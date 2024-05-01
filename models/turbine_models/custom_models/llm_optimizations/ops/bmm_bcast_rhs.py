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
    "bmm_bcast_rhs",
]


@CustomOp.register(library=LIBRARY)
class bmm_bcast_rhs(CustomOp):
    """Performs a floating point matmul of an 'a' and transposed 'b' tensor.

    The types need not match: the bT tensor will be cast to the dtype of the
    'a' tensor.
    a_desc = ksel.arg_tensor(0)  # Shape bsz, num_head, num_group, kv_len, kv_len
    b_desc = ksel.arg_tensor(1)  # Shape bsz, num_head, kv_len, hidden_dim
    To compute (attn * v) for prefill and decoede:
         m == dyn, k == dyn, n == static
    """

    signature = "bmm_bcast_rhs(Tensor a, Tensor b) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        a_desc = ksel.arg_tensor(0)  # Shape b0, b1, bcast, m, k
        b_desc = ksel.arg_tensor(1)  # Shape b0, b1, k, n
        *a_batch_dims, bcast, a_m, a_k = a_desc.t.shape
        *b_batch_dims, b_k, b_n = b_desc.t.shape
        torch._check(
            a_desc.t.dtype.is_floating_point and b_desc.t.dtype.is_floating_point,
            lambda: f"bmm_bcast_rhs: Expected floating point",
        )
        torch._check(
            a_batch_dims == b_batch_dims,
            lambda: f"bmm_bcast_rhs arg 'a': Expected matching batch dims (got {a_batch_dims} vs {b_batch_dims})",
        )
        torch._check(
            len(a_batch_dims) == 2,
            lambda: f"bmm_bcast_rhs arg 'a': Expected 2d batch (got {a_batch_dims})",
        )
        torch._check(
            len(a_desc.t.shape) == 5,
            f"bmm_bcast_rhs arg 'a': Expected 5d tensor (got {a_desc.t.shape})",
        )
        torch._check(
            len(b_desc.t.shape) == 4,
            f"bmm_bcast_rhs arg 'b': Expected 4d tensor (got {b_desc.t.shape})",
        )
        torch._check(
            a_k == b_k,
            f"bmm_bcast_rhs arg 'bT': Expected matching K dimension ({a_k} vs {b_k})",
        )

        # Specialize on the k and n dims.
        a_desc.specialize_dims(0)
        a_desc.specialize_dims(1)
        a_desc.specialize_dims(2)

        b_desc.specialize_dims(0)
        b_desc.specialize_dims(1)
        b_desc.specialize_dims(3)

        # Result 0: Shape b0, b1, bcast, m, n
        out_desc = ksel.return_new_tensor(
            a_batch_dims + [bcast, a_m, b_n], a_desc.t.dtype
        )
        out_desc.specialize_dims(0)
        out_desc.specialize_dims(1)
        out_desc.specialize_dims(2)
        out_desc.specialize_dims(4)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        a = kb.arg_value(0)
        b = kb.arg_value(1)
        a_tensor_type = RankedTensorType(a.type)
        rank = a_tensor_type.rank
        b0, b1, bcast, m, _ = a_tensor_type.shape
        b_tensor_type = RankedTensorType(b.type)
        *_, n = b_tensor_type.shape
        a_type_str = str(a_tensor_type.element_type)
        b_type_str = str(b_tensor_type.element_type)
        try:
            precision_type_str = _PRECISION_TYPE[f"{a_type_str}{b_type_str}"]
        except KeyError:
            raise ValueError(
                f"bmm_bcast_rhs: No accumulator type defined for {a_type_str} x {b_type_str}"
            )
        kwargs = {
            "b0": b0,
            "b1": b1,
            "bcast": bcast,
            "n": n,
            "a_type": a_type_str,
            "b_type": b_type_str,
            "precision": precision_type_str,
        }
        base._JINJA2_ENVIRONMENT = Environment(
            loader=PackageLoader(__name__, "templates")
        )
        template_file = "bmm_bcast_rhs.mlir"
        target_function_name = (
            f"sharktank_bmm_bcast_rhs_deq_{b0}_{b1}_{bcast}_{n}_{precision_type_str}"
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
