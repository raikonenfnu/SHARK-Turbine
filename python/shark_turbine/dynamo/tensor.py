# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A Turbine tensor.

This implementation is adapted from a variety of sources, most notably the subclass
zoo: https://github.com/albanD/subclass_zoo/blob/main/new_device.py
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import functools
import sys
import atexit
from array import array
import numpy as np

import torch
import torch._dynamo as dynamo
from torch.overrides import TorchFunctionMode

from .device import (
    Device,
    DeviceState,
)
from .executor import (
    EagerSpecializedExecutable,
)

from ..support import (
    ApiSequencingError,
    UnknownDTypeError,
)

from iree.runtime import (
    HalBuffer,
    HalBufferView,
    HalCommandBuffer,
    HalElementType,
    HalFence,
    VmModule,
)

from iree.compiler.api import Session, Output
from iree.compiler.passmanager import PassManager

from .importer import FxImporter
from .passes import turbine_cpu_pass_pipeline

DEFAULT_COMPILER_FLAGS = (
    # Enable asynchronous calling convention.
    # TODO: Enable async execution mode.
    # "--iree-execution-model=async-external",
    "--iree-input-type=tm_tensor",
)


###############################################################################
# Factories and device enablement
###############################################################################


class TurbineMode(TorchFunctionMode):
    """Enables PyTorch tensor device= support for Tensor factory functions.

    This can be used in a `with` block to dynamically scope enablement, or
    it can be enabled globally via the `enable()` function.
    """

    IMPLEMENTATIONS = {}
    COMPUTE_METHODS = set()
    CACHED_IMPLEMENTATIONS = {}

    def __torch_function__(self, func, types, args=(), kwargs={}):
        def super_fn(*args, **kwargs):
            # Disable torch_function by hand because we don't want the wrapping behavior of
            # the super() impl
            with torch._C.DisableTorchFunction():
                return func(*args, **kwargs)
        if func in self.IMPLEMENTATIONS:
            if func in self.COMPUTE_METHODS:
                args += (func,)
            return self.IMPLEMENTATIONS[func](super_fn, *args, **kwargs or {})

        # This is just a no-op for all the non-factory functions:
        return super_fn(*args, **kwargs or {})


def enable():
    """Enables PyTorch tensor device= support for Turbine permanently."""
    TurbineMode().__enter__()
    Device("local-task").set()
    atexit.register(disable)

def disable():
    Device.current().clear()
    TurbineMode().__exit__(None, None, None)

# Convenient wrapper to register functions
def raw_factory(func):
    """Decorator to register an unconditional factory function."""

    def _inner_fn(impl):
        TurbineMode.IMPLEMENTATIONS[func] = impl
        return impl

    return _inner_fn

# Convenient wrapper to register functions
def compute_factory(func):
    """Decorator to register an unconditional factory function."""

    def _inner_fn(impl):
        TurbineMode.IMPLEMENTATIONS[func] = impl
        TurbineMode.COMPUTE_METHODS.add(func)
        return impl

    return _inner_fn


def device_factory(func):
    """Decorator to invoke the user provided factory for our devices.

    Wrap a function like this:

    @device_factory(torch.zeros)
    def _zeros(*args, device: Device, **kwargs):
        ...
    """

    def _inner_fn(impl):
        def _filter_impl(super_fn, *args, **kwargs):
            device: Optional[Device] = None
            device_spec = kwargs.get("device", None)
            if device_spec:
                device = _parse_device(device_spec)
            if device:
                del kwargs["device"]
                return impl(*args, device=device, **kwargs)
            return super_fn(*args, **kwargs)

        TurbineMode.IMPLEMENTATIONS[func] = _filter_impl

    return _inner_fn


_TURBINE_PREFIX = "turbine-"


def _parse_device(device_arg) -> Optional[Device]:
    if isinstance(device_arg, Device):
        return device_arg
    elif isinstance(device_arg, str):
        if device_arg == "turbine":
            return Device.current()
        elif device_arg.startswith(_TURBINE_PREFIX):
            return Device(device_arg[len(_TURBINE_PREFIX) :])


###############################################################################
# Turbine storage
###############################################################################


class Storage:
    __slots__ = [
        "buffer",
        "device",
        "ready_fence",
    ]

    def __init__(self, device: Device, buffer: HalBuffer):
        fence_capacity = device._fence_capacity
        self.buffer = buffer
        self.device = device
        # Signalled when the buffer is ready to be consumed. Consumers should
        # join this fence and wait on it. It must be advanced when dependencies
        # are queued.
        self.ready_fence = HalFence(fence_capacity)

    def sync(self):
        """Stops the world and waits for all scheduled mutations to complete."""
        self.ready_fence.wait()

    def execute_transfer(self, cb: HalCommandBuffer):
        """Executes a transfer command buffer that has no external dependencies."""
        device = self.device
        hal_device = device.hal_device
        device._tx_timepoint += 1
        signal_sem = (device._tx_timeline, device._tx_timepoint)
        hal_device.queue_execute(
            [cb], wait_semaphores=self.ready_fence, signal_semaphores=[signal_sem]
        )
        self.ready_fence.insert(*signal_sem)

    def kill(self):
        """Kills the device memory associated with this storage."""
        if not self.buffer:
            raise ApiSequencingError("Storage.kill() called on a non-live instance")
        device = self.device
        hal_device = device.hal_device
        hal_device.queue_dealloca(self.buffer, self.ready_fence, [])
        self.buffer = None
        self.device = None

    def __del__(self):
        if self.buffer:
            self.kill()


###############################################################################
# Tensor class and support
###############################################################################


class DeviceTensor(torch.Tensor):
    """A Tensor accessing memory on a Turbine device."""

    def __torch_dispatch__(cls, func, types, args=(), kwargs={}):
        # Now, we check the function to determine how to handle it. If it's
        # aten.add, then we call aten.sub. Otherwise, we pass through to
        # the original function
        args += (func,)
        return compute_method(func, *args, **kwargs)

    def _to_meta_tensor(self):
        return torch.empty(self.shape, dtype=self.dtype)

    @staticmethod
    def __new__(cls, size, dtype, raw_data=None, requires_grad=False):
        # Using a meta tensor as the wrapped gives us shape and dtype
        # propagation.
        return torch.Tensor._make_subclass(
            cls,
            torch.empty(size, dtype=dtype, device="meta"),
            require_grad=requires_grad,
        )

    def __init__(self, size, dtype, raw_data=None, requires_grad=False):
        if isinstance(raw_data, Storage):
            self._storage = raw_data
            self._bv = None
        else:
            if raw_data is not None:
                raise NotImplementedError(
                    f"raw_data= not implemented for DeviceTensor ({raw_data.__class__})"
                )
    @staticmethod
    def from_torch(input_tensor : torch.Tensor):
        if isinstance(input_tensor, torch.Tensor):
            dev_tensor = DeviceTensor._async_create_empty(input_tensor.size(), Device("local-task"), input_tensor.dtype)
            dev_tensor._async_copy_from_host(input_tensor.numpy())
            return dev_tensor
        else:
            if input_tensor is not None:
                raise ValueError("Expected input to be of type torch.Tensor.")

    @property
    def buffer_view(self) -> HalBufferView:
        if self._bv is None:
            self._bv = HalBufferView(
                self._storage.buffer,
                shape=self.size(),
                element_type=_dtype_to_element_type(self.dtype),
            )
        return self._bv

    def cpu(self):
        return self.to("cpu")

    @property
    def device(self):
        return "turbine"

    def __repr__(self):
        hal_device = self._storage.device.hal_device
        try:
            return f"<TurbineTensor(Device) of {self.buffer_view} on {hal_device}>"
        except UnknownDTypeError:
            return f"<TurbineTensor(Device) of invalid dtype at {self._storage.buffer} on {hal_device}>"

    @staticmethod
    def _async_create_empty(
        size: Sequence[int], device: Device, dtype: torch.dtype
    ) -> "DeviceTensor":
        """Creates an uninitialized tensor with a given size and dtype."""
        alloc_size = _calculate_c_contig_size(size, dtype)
        hal_device = device.hal_device
        # Async allocate a buffer, waiting for the device (tx_timeline, tx_timepoint)
        # and signalling tx_timepoint + 1. Because we are just creating an empty
        # (uninitialized) tensor, it is ready when allocation completes.
        tx_semaphore = device._tx_timeline
        current_tx_timepoint = device._tx_timepoint
        wait_semaphores = [(tx_semaphore, current_tx_timepoint)]
        alloca_complete_semaphore = (tx_semaphore, current_tx_timepoint + 1)
        signal_semaphores = [alloca_complete_semaphore]
        device._tx_timepoint += 1
        buffer = hal_device.queue_alloca(alloc_size, wait_semaphores, signal_semaphores)
        storage = Storage(device, buffer)
        storage.ready_fence.insert(*alloca_complete_semaphore)
        return DeviceTensor(size, dtype, raw_data=storage)

    def _async_fill_py_value(self, value):
        """Fills a value in all elements of the tensor.

        The value is interpreted relative to the tensor's dtype and is suitable for integer
        values like 0, 1, etc. Anything more complicated should use a lower-level API to
        set up a fill pattern.
        """
        storage = self._storage
        hal_device = storage.device.hal_device
        cb = HalCommandBuffer(hal_device)
        pattern = _create_pattern_for_dtype(self.dtype, value)
        cb.fill(storage.buffer, pattern, end=True)
        storage.execute_transfer(cb)

    def _async_copy_from_host(self, host_data):
        """Copies from arbitrary host data of unknown providence.

        Note that this is pretty much the worst way to get data onto the device as
        the default path for many devices involves either host copies or expensive
        device synchronization in order to setup memory mappings. However, as a
        general purpose fallback, its utility cannot be denied.
        """
        storage = self._storage
        hal_device = storage.device.hal_device
        staging_buffer = hal_device.allocator.allocate_host_staging_buffer_copy(
            hal_device, host_data
        )
        cb = HalCommandBuffer(hal_device)
        cb.copy(staging_buffer, storage.buffer, end=True)
        storage.execute_transfer(cb)


def _normalize_size(size_or_nested) -> Sequence[int]:
    if len(size_or_nested) == 1 and not isinstance(size_or_nested[0], int):
        return size_or_nested[0]
    else:
        return size_or_nested


def _calculate_c_contig_size(size: Sequence[int], dtype: torch.dtype) -> int:
    """Calculates a C-contiguous buffer size in bytes for torch size and dtype."""
    accum = _DTYPE_TO_ELEMENT_SIZE[dtype]
    for s in size:
        accum *= s
    return accum


# And some factory functions
# By hand
@raw_factory(torch.Tensor.to)
def to(super_fn, self, device, dtype=None, non_blocking=None):
    # Note that we only implement a subset of .to() here
    turbine_device = _parse_device(device)
    if turbine_device:
        # To turbine.
        # For now, falling back to a copy via CPU.
        new_t = DeviceTensor._async_create_empty(
            self.size(), turbine_device, self.dtype
        )
        new_t._async_copy_from_host(self.numpy())
        return new_t
    elif isinstance(self, DeviceTensor):
        # From turbine.
        # TODO: We can handle certain catwalk cases from/to specific device classes
        # before just falling back to transferring through the CPU.
        # Stop the world and transfer to CPU.
        storage = self._storage
        storage.sync()
        bv = self.buffer_view
        dtype_descr = HalElementType.map_to_dtype(bv.element_type)
        memory = storage.buffer.map()
        np_array = memory.asarray(self.size(), dtype_descr)
        return torch.from_numpy(np_array)
    else:
        return super_fn(self, device)

@raw_factory(torch._C._nn._parse_to)
def _parse_to(super_fn, *args, **kwargs):
    if "turbine" in args:
      #TODO: Parse through args and kwargs for correct params.
      device = "turbine"
      dtype = None
      non_blocking = False
      convert_to_format = None
      return device, dtype, non_blocking, convert_to_format
    else:
        return super_fn(self, device)

@device_factory(torch.empty)
def _empty(*size, device: Device, dtype=torch.float32):
    # Turbine empty.
    size = _normalize_size(size)
    return DeviceTensor._async_create_empty(size, device=device, dtype=dtype)


@device_factory(torch.zeros)
def _zeros(*size, device: Device, dtype=torch.float32):
    t = DeviceTensor._async_create_empty(_normalize_size(size), device, dtype)
    t._async_fill_py_value(0)
    return t


@device_factory(torch.ones)
def _ones(*size, device: Device, dtype=torch.float32):
    t = DeviceTensor._async_create_empty(_normalize_size(size), device, dtype)
    t._async_fill_py_value(1)
    return t


def cpu_tensor_constructor(cpu_func):
    """For our devices, calls a user function which returns a CPU tensor.

    The returned CPU tensor will
    The contents of the array will be copied to a new empty tensor.
    While not terribly efficient, this can be used to fill in bulk-factory
    functions that have not yet been optimized to run completely on device.
    """

    def inner(*args, device: Device, **kwargs):
        cpu_t = cpu_func(*args, **kwargs)
        dev_t = DeviceTensor._async_create_empty(cpu_t.size(), device, cpu_t.dtype)
        dev_t._async_copy_from_host(cpu_t.numpy())
        return dev_t

    return inner


@device_factory(torch.arange)
@cpu_tensor_constructor
def _arange(*args, dtype=None):
    if dtype is not None:
        dtype = _torch_dtype_to_numpy(dtype)
    return torch.from_numpy(np.arange(*args, dtype=dtype))


@device_factory(torch.rand)
@cpu_tensor_constructor
def _rand(*args, dtype=None):
    t = torch.from_numpy(np.random.rand(*args))
    if dtype:
        t = t.to(dtype)
    return t


@functools.lru_cache(maxsize=None)
def _get_device_state() -> DeviceState:
    return DeviceState(driver="local-task")

# Inspiration from https://github.com/nod-ai/SHARK-Turbine/blob/8293de5414889c72ff5cd10bf33c43fb0a3ea3ee/python/shark_turbine/aot/builtins/jittable.py#L212-L237
# and https://github.com/nod-ai/SHARK-Turbine/blob/main/python/shark_turbine/dynamo/backends/cpu.py
# TODO: Try to generalize for other devices.
@compute_factory(torch.add)
@compute_factory(torch.sub)
@compute_factory(torch.mul)
@compute_factory(torch.abs)
def compute_method(super_fn, *args, **kwargs):
    # Compute factory fns reserve the last arg as src_op
    # Requires src_op rather than super_fn, because super_fn
    # is often wrapped by DisableTorchFunction.
    py_args = args[:-1]
    src_op = args[-1]

    # Check if turbine and if all devices are the same.
    py_args =[DeviceTensor.from_torch(torch.tensor(py_arg)) if isinstance(py_arg, (int, float)) else py_arg for py_arg in py_args]
    devices = {tensor.device for tensor in py_args}
    if "turbine" not in devices:
        super_fn(*py_args, **kwargs)
    if len(devices) > 1:
        raise ValueError("Turbine do not support mixed device!")
    # Get a unique encoding to identify computation/dispatch using opCode, input shapes, and dtypes.
    compute_id_encode = str(src_op) + "".join([str(py_arg.shape) + str(py_arg.dtype) for py_arg in py_args])
    compute_hash = hash(compute_id_encode)
    if compute_hash in TurbineMode.CACHED_IMPLEMENTATIONS:
        # TODO: Handle multiple output.
        return DeviceTensor.from_torch(TurbineMode.CACHED_IMPLEMENTATIONS[compute_hash](*py_args, **kwargs)[0])

    # Preprocess func and generate into FX.
    flat_pytorch_args = [py_arg._to_meta_tensor() for py_arg in py_args]
    # TODO: Replace all the below with torch.compile, although currently seems like
    #       the problem lies in it will try to generate DeviceTensor, but it would be missing
    #       _storage and causes error.
    def func_src_op(*args, **kwargs):
        return src_op(*args, **kwargs)
    exported_f = dynamo.export(
        func_src_op,
        aten_graph=True,
        decomposition_table={},
        constraints={},
        assume_static_by_default=True,
    )
    gm, guards = exported_f(*flat_pytorch_args)

    # Setup mlir compilation pipeline.
    session = Session()
    session.set_flags(*DEFAULT_COMPILER_FLAGS)
    session.set_flags("--iree-hal-target-backends=llvm-cpu")
    context = session.context

    # Generate MLIR from FX.
    importer = FxImporter(context=context)
    module = importer.module
    inv = session.invocation()
    # TODO: Should capture diagnostics.
    inv.enable_console_diagnostics()
    inv.import_module(module.operation)
    importer.import_graph_module(gm)
    with context:
        pm = PassManager.parse("builtin.module(torch-to-iree)")
        pm.run(module.operation)

    # Compile MLIR to vmfb.
    inv.execute()
    output = Output.open_membuffer()
    inv.output_vm_bytecode(output)

    # Map VMFB to buffer.
    device_state = _get_device_state()
    vmfb_module = VmModule.copy_buffer(
        device_state.instance,
        output.map_memory(),
    )
    output.close()

    # Load and execute VMFB file.
    exec = EagerSpecializedExecutable(vmfb_module, device_state)
    res_host = exec(*py_args)

    TurbineMode.CACHED_IMPLEMENTATIONS[compute_hash] = exec

    # Rewrap torch tensor into DeviceTensor and return.
    # TODO: Handle multiple output.
    # TODO: Refactor to not need to create new buffers every time once https://github.com/openxla/iree/pull/14997 lands.
    dev_res = DeviceTensor._async_create_empty(res_host[0].size(), Device("local-task"), res_host[0].dtype)
    dev_res._async_copy_from_host(res_host[0].numpy())
    return dev_res


###############################################################################
# Conversions
###############################################################################

_DTYPE_TO_ELEMENT_TYPE: Dict[torch.dtype, HalElementType] = {
    torch.float16: HalElementType.FLOAT_16,
    torch.bfloat16: HalElementType.BFLOAT_16,
    torch.float32: HalElementType.FLOAT_32,
    torch.float64: HalElementType.FLOAT_64,
    torch.uint8: HalElementType.UINT_8,
    torch.int8: HalElementType.SINT_8,
    torch.int16: HalElementType.SINT_16,
    torch.int32: HalElementType.SINT_32,
    torch.int64: HalElementType.SINT_64,
    torch.bool: HalElementType.BOOL_8,
    torch.qint8: HalElementType.OPAQUE_8,
    torch.quint8: HalElementType.OPAQUE_8,
    torch.complex64: HalElementType.COMPLEX_64,
    torch.complex128: HalElementType.COMPLEX_128,
}


def _dtype_to_element_type(dtype) -> HalElementType:
    try:
        return _DTYPE_TO_ELEMENT_TYPE[dtype]
    except KeyError:
        raise UnknownDTypeError(dtype)


_TORCH_DTYPE_TO_NUMPY = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.bool: np.bool_,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}


def _torch_dtype_to_numpy(torch_dtype: torch.dtype) -> Any:
    try:
        return _TORCH_DTYPE_TO_NUMPY[torch_dtype]
    except KeyError:
        raise UnknownDTypeError(torch_dtype)


_ELEMENT_TYPE_TO_NUMPY_DTYPE = {
    HalElementType.FLOAT_16: np.float16,
    HalElementType.FLOAT_32: np.float32,
    HalElementType.FLOAT_64: np.float64,
    HalElementType.UINT_8: np.uint8,
    HalElementType.SINT_8: np.int8,
    HalElementType.SINT_16: np.int16,
    HalElementType.SINT_32: np.int32,
    HalElementType.SINT_64: np.int64,
    HalElementType.BOOL_8: np.bool_,
    HalElementType.COMPLEX_64: np.complex64,
    HalElementType.COMPLEX_128: np.complex128,
}


def _element_type_to_numpy_dtype(element_type: HalElementType) -> Any:
    try:
        return _DTYPE_TO_ELEMENT_TYPE[element_type]
    except KeyError:
        raise UnknownDTypeError(element_type)


def _create_pattern_for_dtype(dtype: torch.dtype, x):
    ctor = _simple_pattern_ctors.get(dtype, None)
    if ctor:
        return ctor(x)
    else:
        raise UnknownDTypeError(dtype)


_simple_pattern_ctors = {
    torch.float16: lambda x: np.float16(float(x)),
    torch.float32: lambda x: np.float32(float(x)),
    torch.float64: lambda x: np.float64(float(x)),
    torch.uint8: lambda x: np.uint8(int(x)),
    torch.int8: lambda x: np.int8(int(x)),
    torch.int16: lambda x: np.int16(int(x)),
    torch.int32: lambda x: np.int32(int(x)),
    torch.int64: lambda x: np.int64(int(x)),
    torch.bool: lambda x: np.bool_(bool(x)),
    torch.complex64: lambda x: np.complex64(complex(x)),
    torch.complex128: lambda x: np.complex128(complex(x)),
}


# returns the torch datatype element size in bytes
_DTYPE_TO_ELEMENT_SIZE = {
    torch.quint4x2: 1,
    torch.uint8: 1,
    torch.int8: 1,
    torch.quint8: 1,
    torch.qint8: 1,
    torch.int16: 2,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int32: 4,
    torch.qint32: 4,
    torch.float32: 4,
    torch.complex32: 4,
    torch.int64: 8,
    torch.float64: 8,
    torch.complex64: 8,
    torch.complex128: 16,
}
