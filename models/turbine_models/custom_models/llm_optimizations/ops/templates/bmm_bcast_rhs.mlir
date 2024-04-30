// query_states torch.Size([1, 128, 16, 32]) -> torch.Size([1, 8, 16, 16, 32]) bcdmk
// key_states torch.Size([1, 8, 16, 32]) ->  torch.Size([1, 8, 1, 32, 16]
// mm: 1, 8, 16, 16, 16 (bcdmk,bckn -> bcdmn)

// attn_weight: torch.Size([1, 8, 16, 16, 16]) bcdmk
// value_states: torch.Size(([1, 8, 16, 32])) -> torch.Size([1, 8, 1, 16, 32]) bckn
// mm: 1, 8, 16, 16, 32 (bcdmk,bckn -> bcdmn)
// module {
//     util.func private @bmm_bcast_rhs(%lhs: tensor<1x8x16x16x32xf16>,  %rhs: tensor<1x8x32x16xf16>) -> tensor<1x8x16x16x16xf16>{
//         %zero = arith.constant 0.0: f16
//         %empty = tensor.empty() : tensor<1x8x16x16x16xf16>
//         %fill = linalg.fill ins(%zero : f16) outs(%empty : tensor<1x8x16x16x16xf16>) -> tensor<1x8x16x16x16xf16>
//         %result = linalg.generic {
//             indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5)>, 
//             affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>, 
//             affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>],
//             iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]} 
//             ins(%lhs, %rhs : tensor<1x8x16x16x32xf16>, tensor<1x8x32x16xf16>) outs(%fill : tensor<1x8x16x16x16xf16>) {
//         ^bb0(%in: f16, %in_4006: f16, %out: f16):
//         %3501 = arith.mulf %in, %in_4006 : f16
//         %3502 = arith.addf %3501, %out : f16
//         linalg.yield %3502 : f16
//         } -> tensor<1x8x16x16x16xf16>
//         util.return %result : tensor<1x8x16x16x16xf16>
//     }
// }

!float_type = {{float_type}}
!a_tensor_type = tensor<{{b}}x{{h}}x{{g}}x{{m}}x{{k}}x!float_type>
!b_tensor_type = tensor<{{b}}x{{h}}x{{k}}x{{n}}x!float_type>
!c_tensor_type = tensor<{{b}}x{{h}}x{{g}}x{{m}}x{{n}}x!float_type>

module {
    util.func @sharktank_bmm_bcast_rhs_{{b}}_{{h}}_{{g}}_{{m}}_{{n}}_{{k}}_{{float_type}}(%lhs: !a_tensor_type,  %rhs: !b_tensor_type) -> !c_tensor_type {
        %zero = arith.constant 0.0: !float_type
        %empty = tensor.empty() : !c_tensor_type
        %fill = linalg.fill ins(%zero : !float_type) outs(%empty : !c_tensor_type) -> !c_tensor_type
        %result = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5)>, 
            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>, 
            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]} 
            ins(%lhs, %rhs : !a_tensor_type, !b_tensor_type) outs(%fill : !c_tensor_type) {
        ^bb0(%in: !float_type, %in_4006: !float_type, %out: !float_type):
        %3501 = arith.mulf %in, %in_4006 : !float_type
        %3502 = arith.addf %3501, %out : !float_type
        linalg.yield %3502 : !float_type
        } -> !c_tensor_type
        util.return %result : !c_tensor_type
    }
}

// module {
//     util.func @sharktank_bmm_bcast_rhs(%lhs: tensor<1x8x8x1x?xf16>,  %rhs: tensor<1x8x?x128xf16>) -> tensor<64x1x128xf16> {
//         %zero = arith.constant 0.0: f16
//         %empty = tensor.empty() : tensor<1x8x8x1x128xf16>
//         %fill = linalg.fill ins(%zero : f16) outs(%empty : tensor<1x8x8x1x128xf16>) -> tensor<1x8x8x1x128xf16>
//         %result = linalg.generic {
//             indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5)>, 
//             affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>, 
//             affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>],
//             iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]} 
//             ins(%lhs, %rhs : tensor<1x8x8x1x?xf16>, tensor<1x8x?x128xf16>) outs(%fill : tensor<1x8x8x1x128xf16>) {
//         ^bb0(%in: f16, %in_4006: f16, %out: f16):
//         %3501 = arith.mulf %in, %in_4006 : f16
//         %3502 = arith.addf %3501, %out : f16
//         linalg.yield %3502 : f16
//         } -> tensor<1x8x8x1x128xf16>
//         %collapsed_result = tensor.collapse_shape %result [[0, 1, 2], [3], [4]] : tensor<1x8x8x1x128xf16> into tensor<64x1x128xf16>
//         util.return %collapsed_result : tensor<64x1x128xf16>
//     }
// }