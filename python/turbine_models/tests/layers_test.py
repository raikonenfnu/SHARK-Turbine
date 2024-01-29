import torch
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.modeling_llama import LlamaConfig
import unittest

def main():
    batch_size = 1
    seq_len = 16
    hidden_size = 1024
    vocab_size = 3200
    config = LlamaConfig()
    config.vocab_size = vocab_size
    config.hidden_size = hidden_size
    llama_attn_layer = LlamaAttention(config, 0)
    sample_input = torch.randn([batch_size, seq_len, hidden_size])
    
    print(llama_attn_layer.forward(sample_input))

if __name__ == "__main__":
    main()