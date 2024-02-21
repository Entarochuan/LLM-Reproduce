"""
    demo configs 
    
"""

model_args = dict(
    hidden_size = 512,
    n_layers = 8,
    n_heads = 8,
    vocab_size = 32000,  # 使用mistral tokenizer
    multiple_of = 128,  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps = 1e-5,
    use_RMS = True,
    use_RoPE = True,
    max_batch_size = 32,
    max_seq_len = 2048,
)