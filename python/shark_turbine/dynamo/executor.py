# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
from typing import List, Optional, Sequence, Union
import torch
from collections import namedtuple

from iree.runtime import (
    asdevicearray,
    create_hal_module,
    HalBufferView,
    HalElementType,
    DeviceArray,
    get_driver,
    VmContext,
    HalDevice,
    HalDriver,
    HalFence,
    VmInstance,
    VmModule,
    VmVariantList,
)

from torch import (
    from_numpy as torch_from_numpy,
)

from .device import Device, DeviceState
NUMPY_STR_TO_TORCH_DTYPE = {
    "uint8" : torch.uint8,
    "int8" : torch.int8,
    "int16" : torch.int16,
    "int32" : torch.int32,
    "int64" : torch.int64,
    "float16" : torch.float16,
    "float32" : torch.float32,
    "float64" : torch.float64,
    "bool_" : torch.bool,
}


@functools.lru_cache(maxsize=None)
def get_vm_instance() -> VmInstance:
    return VmInstance()


class SpecializedExecutable:
    """A concrete executable that has been specialized in some way."""

    __slots__ = [
        "device_state",
        "entry_function",
        "user_module",
        "vm_context",
    ]

    def __init__(
        self,
        user_module: VmModule,
        device_state: DeviceState,
        entry_name: str = "main",
    ):
        self.user_module = user_module
        self.vm_context = VmContext(
            device_state.instance,
            (
                create_hal_module(device_state.instance, device_state.device),
                user_module,
            ),
        )
        self.device_state = device_state
        self.entry_function = self.user_module.lookup_function(entry_name)

    def __call__(self, *inputs):
        arg_list = VmVariantList(len(inputs))
        ret_list = VmVariantList(
            1
        )  # TODO: Get the number of results from the descriptor.

        # Move inputs to the device and add to arguments.
        self._inputs_to_device(inputs, arg_list)
        # TODO: Append semaphores for async execution.

        # Invoke.
        self.vm_context.invoke(self.entry_function, arg_list, ret_list)
        return self._returns_to_user(ret_list)

    def _inputs_to_device(self, inputs: list, arg_list: VmVariantList):
        # TODO: We are assuming the worst case here which is that we have unknown Torch
        # tensors that we send to the CPU and make continguous. Ideally, we would have
        # fast paths for our own backends and interop.
        for input in inputs:
            input_cpu = input.cpu().contiguous()
            # Since this is already a fallback case, just use the numpy array interop.
            # It isn't great, but meh... fallback case.
            device_array = asdevicearray(self.device_state.device, input_cpu)
            arg_list.push_ref(device_array._buffer_view)

    def _returns_to_user(self, ret_list: VmVariantList):
        # TODO: This is also not good that we are moving back to the CPU like this.
        # We should be returning a custom Tensor implementation which represents
        # our device data and has synchronization hooks for accessing it.
        device = self.device_state.device
        num_returns = len(ret_list)
        user_returns = [None] * num_returns
        for i in range(num_returns):
            device_buffer_view = HalBufferView.__iree_vm_cast__(ret_list.get_as_ref(i))
            device_array = DeviceArray(device, device_buffer_view)
            host_array = device_array.to_host()
            user_returns[i] = torch_from_numpy(host_array)

        return user_returns



AsyncResult = namedtuple('AsyncResult', ['buffer', 'size', 'dtype', 'signal'])

class EagerSpecializedExecutable:
    """A concrete executable that has been specialized in some way."""

    __slots__ = [
        "device_state",
        "entry_function",
        "user_module",
        "vm_context",
    ]

    def __init__(
        self,
        user_module: VmModule,
        device_state: DeviceState,
        entry_name: str = "main",
    ):
        self.user_module = user_module
        self.vm_context = VmContext(
            device_state.instance,
            (
                create_hal_module(device_state.instance, device_state.device),
                user_module,
            ),
        )
        self.device_state = device_state
        self.entry_function = self.user_module.lookup_function(entry_name)

    def __call__(self, *inputs):
        arg_list = VmVariantList(len(inputs))
        ret_list = VmVariantList(
            1
        )  # TODO: Get the number of results from the descriptor.

        # Move inputs to the device and add to arguments.
        self._inputs_to_device(inputs, arg_list)
        # TODO: Append semaphores for async execution.

        # Invoke.
        # TODO: Assert that exec need at least 1 input.
        # TODO: Think about better way to get device.
        device = inputs[0]._storage.device
        fence_capacity = device._fence_capacity
        tx_semaphore = device._tx_timeline
        current_tx_timepoint = device._tx_timepoint

        # Create wait semaphore and fence.
        wait_semaphores = (tx_semaphore, current_tx_timepoint)
        wait_fence = HalFence(fence_capacity)
        wait_fence.insert(*wait_semaphores)

        # Create signal semaphore and fence.
        signals_semaphore = (tx_semaphore, current_tx_timepoint + 1)
        signal_fence = HalFence(fence_capacity)
        signal_fence.create_at(*signals_semaphore)

        # Add fences into arg_list for async exec.
        arg_list.push_ref(wait_fence)
        arg_list.push_ref(signal_fence)
        device._tx_timepoint += 1
        self.vm_context.invoke(self.entry_function, arg_list, ret_list)
        return self._returns_to_user(ret_list, signal_fence)

    def _inputs_to_device(self, inputs: list, arg_list: VmVariantList):
        # TODO: We are assuming the worst case here which is that we have unknown Torch
        # tensors that we send to the CPU and make continguous. Ideally, we would have
        # fast paths for our own backends and interop.
        for input in inputs:
            arg_list.push_ref(input.buffer_view)

    def _returns_to_user(self, ret_list: VmVariantList, signal_fence: HalFence):
        # TODO: This is also not good that we are moving back to the CPU like this.
        # We should be returning a custom Tensor implementation which represents
        # our device data and has synchronization hooks for accessing it.
        device = self.device_state.device
        num_returns = len(ret_list)
        user_returns = [None] * num_returns
        for i in range(num_returns):
            device_buffer_view = HalBufferView.__iree_vm_cast__(ret_list.get_as_ref(i))
            npy_dtype = HalElementType.map_to_dtype(device_buffer_view.element_type)
            size = torch.Size(device_buffer_view.shape)
            dtype = NUMPY_STR_TO_TORCH_DTYPE[npy_dtype.name]
            device_buffer = device_buffer_view.get_buffer()
            user_returns[i] = AsyncResult(device_buffer, size, dtype, signal_fence)
        return user_returns

#         tx_semaphore = device._tx_timeline
#         current_tx_timepoint = device._tx_timepoint
#         wait_semaphores = [(tx_semaphore, current_tx_timepoint)]
#         alloca_complete_semaphore = (tx_semaphore, current_tx_timepoint + 1)
#         signal_semaphores = [alloca_complete_semaphore]
#         device._tx_timepoint += 1
#         buffer = hal_device.queue_alloca(alloc_size, wait_semaphores, signal_semaphores)
#         storage = Storage(device, buffer)
#         storage.ready_fence.insert(*alloca_complete_semaphore)
#         return DeviceTensor(size, dtype, raw_data=storage)
