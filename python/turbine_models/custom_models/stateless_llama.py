import os
import sys
import re

os.environ["TORCH_LOGS"] = "dynamic"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils import _pytree as pytree
from shark_turbine.aot import *
from iree.compiler.ir import Context

from turbine_models.custom_models import remap_gguf
import safetensors

BATCH_SIZE = 1
MAX_STEP_SEQ = 4095

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_auth_token", type=str, help="The Hugging Face auth token, required"
)
parser.add_argument("--compile_to", type=str, help="torch, linalg, vmfb")
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="meta-llama/Llama-2-7b-chat-hf",
)
parser.add_argument("--quantization", type=str, default="unquantized")
parser.add_argument("--external_weight_file", type=str, default="")
parser.add_argument("--vmfb_path", type=str, default="")
parser.add_argument(
    "--external_weights",
    type=str,
    default=None,
    help="saves ir/vmfb without global weights for size and readability, options [gguf, safetensors]",
)
parser.add_argument(
    "--precision", type=str, default="fp16", help="dtype of model [f16, f32]"
)

parser.add_argument(
    "--device", type=str, default="llvm-cpu", help="llvm-cpu, cuda, vulkan, rocm"
)
# TODO: Bring in detection for target triple
parser.add_argument(
    "--iree_target_triple",
    type=str,
    default="host",
    help="Specify vulkan target triple or rocm/cuda target device.",
)
parser.add_argument("--vulkan_max_allocation", type=str, default="4294967296")

# TODO (Dan): replace this with a file once I figure out paths on windows exe
json_schema = """
[1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}]}]
"""


def slice_up_to_step(global_pkv, seq_step, heads, hidden_dim):
    all_pkv_tensors = []
    for i in range(heads * 2):
        # Numpy semantic: sliced = global_pkv[i, 0, 0:seq_step, 0:heads, 0:hidden_dim]
        # Generates tensor<1 x 1 x seq_step x heads x hidden_dim>
        sliced = IREE.tensor_slice(
            global_pkv, i, 0, (0, seq_step), (0, heads), (0, hidden_dim)
        )  # sequence context dim
        all_pkv_tensors.append(
            IREE.tensor_reshape(sliced, 1, seq_step, heads, hidden_dim)
        )

    return all_pkv_tensors


def export_transformer_model(
    hf_model_name,
    hf_auth_token=None,
    compile_to="torch",
    external_weights=None,
    external_weight_file=None,
    quantization=None,
    precision=None,
    device=None,
    target_triple=None,
    vulkan_max_allocation=None,
):
    state_schema = pytree.treespec_loads(json_schema)

    mod = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.float,
        token=hf_auth_token,
    )
    dtype = torch.float32
    if precision == "f16":
        mod = mod.half()
        dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        use_fast=False,
        token=hf_auth_token,
    )
    # TODO: generate these values instead of magic numbers
    HEADS = 32
    HIDDEN_DIM = 128
    BATCH_SIZE = 1
    global_pkv = torch.zeros(
        size=(HEADS * 2, BATCH_SIZE, MAX_STEP_SEQ, HEADS, HIDDEN_DIM),
        dtype=dtype,
    )

    mapper = {}
    if external_weights is not None:
        if external_weights == "safetensors":
            mod_params = dict(mod.named_parameters())
            for name in mod_params:
                mapper["params." + name] = name
            if external_weight_file:
                safetensors.torch.save_file(mod_params, external_weight_file)

        elif external_weights == "gguf":
            tensor_mapper = remap_gguf.TensorNameMap(remap_gguf.MODEL_ARCH.LLAMA, HEADS)
            mapper = tensor_mapper.mapping

    class StateUpdateModule(CompiledModule):
        if external_weights:
            params = export_parameters(
                mod, external=True, external_scope="", name_mapper=mapper.get
            )
        else:
            params = export_parameters(mod)
        global_state = export_global(
            abstractify(global_pkv), uninitialized=True, mutable=True
        )
        global_seq_step = export_global(AbstractIndex, mutable=True)

        def run_initialize(self, x=AbstractTensor(BATCH_SIZE, None, dtype=torch.int64)):
            init_const = [x.dynamic_dim(1) < MAX_STEP_SEQ]
            token, *state = self.initialize(x, constraints=init_const)
            self.global_seq_step = IREE.tensor_dim(
                state[0], 1
            )  # ? dimension of arbitrarily 0th kv tensor
            for i in range(HEADS * 2):
                slice_of_state = IREE.tensor_reshape(
                    state[i], 1, 1, self.global_seq_step, HEADS, HIDDEN_DIM
                )
                self.global_state = IREE.tensor_update(
                    self.global_state, slice_of_state, i, 0, 0, 0, 0
                )
            return token

        def run_cached_initialize(self, x=AbstractTensor(BATCH_SIZE, None, dtype=torch.int64)):
            state_arg = slice_up_to_step(
                self.global_state, self.global_seq_step, HEADS, HIDDEN_DIM
            )
            forw_const = (
                [x.dynamic_dim(1) < MAX_STEP_SEQ]
                + [state_arg[0].dynamic_dim(1) < MAX_STEP_SEQ]
                + [
                    x.dynamic_dim(1) == (state_arg[0].dynamic_dim(1))
                    for x in state_arg[1:]
                ]
                + [x.dynamic_dim(1) < MAX_STEP_SEQ for x in state_arg[1:]]
            )
            token, *state = self.cached_initialize(x, *state_arg, constraints=forw_const)
            len_of_new_tokens = IREE.tensor_dim(
                state[0], 1
            )  # ? dimension of arbitrarily 0th kv tensor
            for i in range(HEADS * 2):
                slice_of_state = IREE.tensor_reshape(
                    state[i], 1, 1, len_of_new_tokens, HEADS, HIDDEN_DIM
                )
                self.global_state = IREE.tensor_update(
                    self.global_state, slice_of_state, i, 0, self.global_seq_step, 0, 0
                )
            self.global_seq_step = self.global_seq_step + len_of_new_tokens
            return token

        # Streaming-LLM KVCache evict algorithm:
        # slice1 = KVCache[0 : sink]
        # slice2 = KVCache[seq_len - window_size : seq_len]
        # KVCache = torch.cat([slice1, slice2])
        # TODO: There is actual overlap of data.
        # For e.g at token length 600, sink size 4, and window size 508
        # Then KVCache[4:512] going to be replaced by KVCache[600-508: (600-508)+508]
        # => KVCache[4:512] = KVCache[92:600] => Much overlap of data(i.e 92->512)
        # => We'd need to do a copy and then replace. Or we can make the gap at least 2X.
        def evict_kvcache_space(self):
            # TODO: Replace hardcoded with global variable.
            sink_size = 4
            window_size = 508
            most_recent_window = self.global_seq_step + (-window_size)
            for i in range(HEADS * 2):
                update_window_state = IREE.tensor_slice(
                    self.global_state, i, 0, (most_recent_window, window_size), (0, HEADS), (0, HIDDEN_DIM)
                )  # sequence context dim
                self.global_state = IREE.tensor_move(
                    self.global_state, update_window_state, i, 0, sink_size, 0, 0
                )
            self.global_seq_step = self.global_seq_step.set(window_size + sink_size)
            return self.global_seq_step

        def run_forward(self, x=AbstractTensor(1, 1, dtype=torch.int64)):
            state_arg = slice_up_to_step(
                self.global_state, self.global_seq_step, HEADS, HIDDEN_DIM
            )
            forw_const = (
                [state_arg[0].dynamic_dim(1) < MAX_STEP_SEQ]
                + [
                    x.dynamic_dim(1) == (state_arg[0].dynamic_dim(1))
                    for x in state_arg[1:]
                ]
                + [x.dynamic_dim(1) < MAX_STEP_SEQ for x in state_arg[1:]]
            )
            token, *state_update = self.forward(x, *state_arg, constraints=forw_const)
            for i in range(HEADS * 2):
                update = IREE.tensor_reshape(
                    state_update[i], 1, 1, 1, HEADS, HIDDEN_DIM
                )
                self.global_state = IREE.tensor_update(
                    self.global_state, update, i, 0, self.global_seq_step, 0, 0
                )

            self.global_seq_step = self.global_seq_step + 1
            return token

        def get_global_state(self):
            return self.global_state

        def get_seq_step(self):
            return self.global_seq_step

        @jittable
        def initialize(input_ids):
            result = mod.forward(input_ids)
            state1_flat, _ = pytree.tree_flatten(result.past_key_values)
            token1 = torch.argmax(result.logits[:, -1, :], dim=1)
            token1 = token1[None, :]
            state1_flat = [torch.transpose(x, 1, 2) for x in state1_flat]
            return token1, *state1_flat

        @jittable
        def cached_initialize(input_ids, *state0_flat):
            # Unpad the states.
            cur_token_len = state0_flat[0].size(1)
            state0_flat = [torch.transpose(x, 1, 2) for x in state0_flat]
            state0 = pytree.tree_unflatten(state0_flat, state_schema)
            result = mod.forward(input_ids, past_key_values=state0)
            state1_flat, _ = pytree.tree_flatten(result.past_key_values)
            state1_flat = [torch.transpose(x[:, :, cur_token_len:, :], 1, 2) for x in state1_flat]
            token1 = torch.argmax(result.logits[:, -1, :], dim=1)
            token1 = token1[None, :]
            return token1, *state1_flat

        @jittable
        def forward(token0: torch.Tensor, *state0_flat):
            # Unpad the states.
            state0_flat = [torch.transpose(x, 1, 2) for x in state0_flat]
            state0 = pytree.tree_unflatten(state0_flat, state_schema)
            result = mod.forward(token0, past_key_values=state0)
            state1_flat, _ = pytree.tree_flatten(result.past_key_values)
            state1_flat = [torch.transpose(x[:, :, -1:, :], 1, 2) for x in state1_flat]
            token1 = torch.argmax(result.logits[:, -1, :], dim=1)
            token1 = token1[None, :]
            return token1, *state1_flat

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = StateUpdateModule(context=Context(), import_to=import_to)
    # TODO: Integrate with external parameters to actually be able to run
    # TODO: Make more generalizable to be able to quantize with all  compile_to options
    if quantization == "int4" and not compile_to == "linalg":
        from shark_turbine.transforms.quantization import mm_group_quant

        mm_group_quant.MMGroupQuantRewriterPass(
            CompiledModule.get_mlir_module(inst).operation
        ).run()
    module_str = str(CompiledModule.get_mlir_module(inst))
    safe_name = hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    if compile_to != "vmfb":
        return module_str, tokenizer
    else:
        flags = [
            "--iree-input-type=torch",
            "--mlir-print-debuginfo",
            "--mlir-print-op-on-diagnostic=false",
            "--iree-llvmcpu-target-cpu-features=host",
            "--iree-llvmcpu-target-triple=x86_64-linux-gnu",
            "--iree-stream-resource-index-bits=64",
            "--iree-vm-target-index-bits=64",
            "--iree-codegen-check-ir-before-llvm-conversion=false",
            "--iree-opt-const-expr-hoisting=False",
        ]
        if device == "cpu" or device == "llvm-cpu":
            flags.append("--iree-llvmcpu-enable-ukernels=all")
            device = "llvm-cpu"
        elif device == "vulkan":
            flags.extend(
                [
                    "--iree-vulkan-target-triple=" + target_triple,
                    "--iree-stream-resource-max-allocation-size="
                    + vulkan_max_allocation,
                ]
            )
        elif device == "rocm":
            flags.extend(
                [
                    "--iree-rocm-target-chip=" + target_triple,
                    "--iree-rocm-link-bc=true",
                    "--iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode",
                    "--iree-vm-bytecode-module-strip-source-map=true",
                    "--iree-opt-strip-assertions=true",
                    "--iree-vm-target-truncate-unsupported-floats",
                ]
            )
        elif device == "cuda":
            flags.extend(
                [
                    "--iree-hal-cuda-llvm-target-arch=" + target_triple,
                    "--iree-vm-bytecode-module-strip-source-map=true",
                    "--iree-vm-target-truncate-unsupported-floats",
                ]
            )
        else:
            print("Unknown device kind: ", device)
        import iree.compiler as ireec

        flatbuffer_blob = ireec.compile_str(
            module_str,
            target_backends=[device],
            extra_args=flags,
        )
        with open(f"{safe_name}.vmfb", "wb+") as f:
            f.write(flatbuffer_blob)
        print("saved to ", safe_name + ".vmfb")
        return module_str, tokenizer


if __name__ == "__main__":
    args = parser.parse_args()
    mod_str, _ = export_transformer_model(
        args.hf_model_name,
        args.hf_auth_token,
        args.compile_to,
        args.external_weights,
        args.external_weight_file,
        args.quantization,
        args.precision,
        args.device,
        args.iree_target_triple,
        args.vulkan_max_allocation,
    )
    safe_name = args.hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to ", safe_name + ".mlir")
