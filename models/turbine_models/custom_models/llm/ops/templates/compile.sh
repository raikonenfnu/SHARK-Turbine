if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

# ~/nod/iree-build-notrace/install/bin/iree-compile $1 \
#   --iree-hal-target-backends=rocm --iree-rocm-target-chip=$2 --iree-codegen-llvmgpu-use-vector-distribution \
#   -o $1_$2.vmfb --mlir-print-ir-after-all 2> $1_$2_dump.mlir

#  --iree-preprocessing-pass-pipeline="builtin.module(iree-preprocessing-pack-to-intrinsics)"
# --iree-codegen-llvmgpu-use-vector-distribution
# --iree-preprocessing-pass-pipeline="builtin.module(iree-preprocessing-pad-to-intrinsics)" --iree-codegen-llvmgpu-use-vector-distribution
# --debug-only=iree-codegen-tile-and-distribute-to-workgroups
~/nod/iree-build-notrace/tools/iree-compile $1 \
	--iree-hal-target-backends=rocm --iree-rocm-target-chip=$2 --iree-preprocessing-pass-pipeline="builtin.module(util.func(iree-preprocessing-pad-to-intrinsics))" --iree-codegen-llvmgpu-use-vector-distribution --mlir-disable-threading --iree-hal-benchmark-dispatch-repeat-count=100 \
  --iree-stream-resource-max-allocation-size=4294967296 --verify=true -o $1_$2.vmfb --debug-only=iree-llvmgpu-vector-distribute --debug-only=iree-codegen-gpu-vector-distribution --mlir-print-ir-after-all 2> $1_$2_dump.mlir
  # --iree-stream-resource-max-allocation-size=4294967296 -o $1_$2.vmfb
  # -o $1_$2.vmfb --debug-only=iree-codegen-tile-and-distribute-to-workgroups 2> $1_$2_tile_and_dist_wg_dynamic.mlir
#   -o $1_$2.vmfb

  # ~/nod/iree-build-notrace/install/bin/iree-compile $1 \
  # --iree-hal-target-backends=llvm-cpu --iree-codegen-llvmgpu-use-vector-distribution --iree-hal-benchmark-dispatch-repeat-count=100 \
  # -o $1_$2.vmfb --debug-only=iree-llvmgpu-vector-distribute --debug-only=iree-codegen-gpu-vector-distribution --mlir-print-ir-after-all 2> $1_$2_dump.mlir
