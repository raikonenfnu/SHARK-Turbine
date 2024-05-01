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
        return ops.bmm_bcast_rhs(a, b)

def main():
    mod = MyModule()
    ep = torch.export.export(
        mod,
        args=(
            torch.rand([1, 8, 16, 128, 128], dtype=torch.float32),
            torch.rand([1, 8, 128, 32], dtype=torch.float32),
        ),
    )
    output = aot.export(ep)
    asm = str(output.mlir_module)

    # e2e test.
    lhs = torch.rand([1, 8, 16, 128, 128], dtype=torch.float32)
    rhs = torch.rand([1, 8, 128, 32], dtype=torch.float32)
    ref = torch.matmul(lhs, rhs.unsqueeze(2))
    result = ops.bmm_bcast_rhs(lhs, rhs)
    torch.testing.assert_close(result, ref)
    print("SUCCESS")
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()

