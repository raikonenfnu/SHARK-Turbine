# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.ops import base
from sharktank.ops.base import *

import os
import torch

__all__ = [
    "bmmt_bcast_rhs",
]


@CustomOp.register(library=LIBRARY)
class bmmt_bcast_rhs(CustomOp):
    """Performs a floating point matmul of an 'a' and transposed 'b' tensor.

    The types need not match: the bT tensor will be cast to the dtype of the
    'a' tensor.
    a_desc = ksel.arg_tensor(0)  # Shape bsz, num_head, num_group, kv_len, hidden_dim
    bT_desc = ksel.arg_tensor(1)  # Shape bsz, num_head, kv_len, hidden_dim
    c_desc = output  # Shape bsz, num_head, kv_len, kv_len
    To compute (q * KT):
        prefill:
            m == dyn, k == static, n == dyn
        decode:
            m == 1, k == static, n == dyn
    """

    signature = "bmmt_bcast_rhs(Tensor a, Tensor b) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        a_desc = ksel.arg_tensor(0)  # Shape b0, b1, bcast, m, k
        bT_desc = ksel.arg_tensor(1)  # Shape b0, b1, n, k
        *a_batch_dims, bcast, a_m, a_k = a_desc.t.shape
        *bT_batch_dims, bT_n, bT_k = bT_desc.t.shape
        torch._check(
            a_desc.t.dtype.is_floating_point and bT_desc.t.dtype.is_floating_point,
            lambda: f"bmmt_bcast_rhs: Expected floating point",
        )
        torch._check(
            a_batch_dims == bT_batch_dims,
            lambda: f"bmmt_bcast_rhs arg 'a': Expected matching batch dims (got {a_batch_dims} vs {bT_batch_dims})",
        )
        torch._check(
            len(a_batch_dims) == 2,
            lambda: f"bmmt_bcast_rhs arg 'a': Expected 2d batch (got {a_batch_dims})",
        )
        torch._check(
            len(a_desc.t.shape) == 5,
            f"bmmt_bcast_rhs arg 'a': Expected 5d tensor (got {a_desc.t.shape})",
        )
        torch._check(
            len(bT_desc.t.shape) == 4,
            f"bmmt_bcast_rhs arg 'b': Expected 4d tensor (got {bT_desc.t.shape})",
        )
        torch._check(
            a_k == bT_k,
            f"bmmt_bcast_rhs arg 'bT': Expected matching K dimension ({a_k} vs {bT_k})",
        )

        # Specialize on the k and n dims.
        # a_desc.specialize_dims(0, 1, 2, 3)
        a_desc.specialize_dims(0)
        a_desc.specialize_dims(1)
        a_desc.specialize_dims(2)
        a_desc.specialize_dims(4)

        # bT_desc.specialize_all_dims(0, 1, 3)
        bT_desc.specialize_dims(0)
        bT_desc.specialize_dims(1)
        bT_desc.specialize_dims(3)

        # Result 0: Shape b0, b1, bcast, m, n
        out_desc = ksel.return_new_tensor(
            a_batch_dims + [bcast, a_m, bT_n], a_desc.t.dtype
        )
        out_desc.specialize_dims(0)
        out_desc.specialize_dims(1)
        out_desc.specialize_dims(2)

        # Specialize m if m ==1. (dequant case)
        # if a_m == 1:
        #     a_desc.specialize_dims(3)
        #     out_desc.specialize_dims(3)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        a = kb.arg_value(0)
        bT = kb.arg_value(1)
        a_tensor_type = RankedTensorType(a.type)
        rank = a_tensor_type.rank
        b0, b1, bcast, _, k = a_tensor_type.shape
        bT_tensor_type = RankedTensorType(bT.type)
        a_type_str = str(a_tensor_type.element_type)
        bT_type_str = str(bT_tensor_type.element_type)
        try:
            precision_type_str = _PRECISION_TYPE[f"{a_type_str}{bT_type_str}"]
        except KeyError:
            raise ValueError(
                f"bmmt_bcast_rhs: No accumulator type defined for {a_type_str} x {bT_type_str}"
            )
        kwargs = {
            "b0": b0,
            "b1": b1,
            "bcast": bcast,
            "k": k,
            "a_type": a_type_str,
            "bT_type": bT_type_str,
            "precision": precision_type_str,
        }
        base._JINJA2_ENVIRONMENT = Environment(
            loader=PackageLoader(__name__, "templates")
        )
        template_file = "bmmt_bcast_rhs.mlir"
        target_function_name = (
            f"sharktank_bmmt_bcast_rhs_deq_{b0}_{b1}_{bcast}_{k}_{precision_type_str}"
        )
        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            **kwargs,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))


_PRECISION_TYPE: dict[str, str] = {
    "f16f16": "f16",
    "f32f32": "f32",
}
