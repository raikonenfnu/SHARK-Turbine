# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from shark_turbine import aot
from sharktank import ops
from turbine_models.custom_models.llm_optimizations import ops

class MyModule(torch.nn.Module):
    def forward(self, a, b):
        return ops.bmmt_bcast_rhs(a, b)

# // query_states torch.Size([1, 128, 16, 32]) -> torch.Size([1, 8, 16, 16, 32]) bcdmk
# // key_states torch.Size([1, 8, 16, 32]) ->  torch.Size([1, 8, 1, 32, 16]

def main():
    mod = MyModule()
    ep = torch.export.export(
        mod,
        args=(
            torch.rand([1, 8, 16, 16, 32], dtype=torch.float32),
            torch.rand([1, 8, 16, 32], dtype=torch.float32),
        ),
    )
    output = aot.export(ep)
    asm = str(output.mlir_module)

    # e2e test.
    lhs = torch.rand([1, 128, 16, 32], dtype=torch.float32)
    rhs = torch.rand([1, 8, 16, 32], dtype=torch.float32)
    num_key_value_heads = rhs.shape[1]
    ref = torch.matmul(
        lhs.reshape(
            [
                lhs.shape[0],
                num_key_value_heads,
                -1,
                *lhs.shape[2:],
            ]
        ),
        rhs.unsqueeze(2).transpose(3, 4),
    )
    result = ops.bmmt_bcast_rhs(lhs.reshape([lhs.shape[0], num_key_value_heads, -1, *lhs.shape[2:]]), rhs)
    torch.testing.assert_close(result, ref)
    print("SUCCESS")
    print(asm)

if __name__ == "__main__":
    main()

