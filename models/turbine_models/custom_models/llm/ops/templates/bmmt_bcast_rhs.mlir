// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


!precision = {{precision}}
!a_tensor_type = tensor<{{b0}}x{{b1}}x{{bcast}}x?x{{k}}x!precision>
!bT_tensor_type = tensor<{{b0}}x{{b1}}x?x{{k}}x!precision>
!c_tensor_type = tensor<{{b0}}x{{b1}}x{{bcast}}x?x?x!precision>

module {
    util.func private @sharktank_bmmt_bcast_rhs_deq_{{b0}}_{{b1}}_{{bcast}}_{{k}}_{{precision}}(%lhs: !a_tensor_type,  %rhs: !bT_tensor_type) -> !c_tensor_type {
        %c2 = arith.constant 2: index
        %c3 = arith.constant 3: index
        %zero = arith.constant 0.0: !precision
        %dim0 = tensor.dim %lhs, %c3 : !a_tensor_type
        %dim1 = tensor.dim %rhs, %c2 : !bT_tensor_type
        %empty = tensor.empty(%dim0, %dim1) : !c_tensor_type
        %fill = linalg.fill ins(%zero : !precision) outs(%empty : !c_tensor_type) -> !c_tensor_type
        %result = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5)>, 
            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>, 
            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]} 
            ins(%lhs, %rhs : !a_tensor_type, !bT_tensor_type) outs(%fill : !c_tensor_type) {
        ^bb0(%in: !precision, %in_4006: !precision, %out: !precision):
        %3501 = arith.mulf %in, %in_4006 : !precision
        %3502 = arith.addf %3501, %out : !precision
        linalg.yield %3502 : !precision
        } -> !c_tensor_type
        util.return %result : !c_tensor_type
    }
}
