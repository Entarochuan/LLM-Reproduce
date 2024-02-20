"""
    Transformer : llama的transformer架构
"""

import torch 
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from multi_head_attention import Multi_Head_Self_Attention, FeedForward
from model_utils import RMSNorm

@dataclass
class ModelArgs:
    hidden_size: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    use_RMS:bool = True
    use_RoPE:bool = True
    
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    
class TransformerBlock(nn.Module):
    
    """
        TransformerBlock (llama style).

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """
    
    def __init__(
        self, 
        layer_id: int, 
        args: ModelArgs
    ) : 
        
        super().__init__()
        self.n_heads = args.n_heads
        self.hidden_size = args.hidden_size
        # self.head_dim = args.hidden_size // args.n_heads
        self.attention = Multi_Head_Self_Attention(self.hidden_size, self.n_heads, args.norm_eps, args.use_RMS, args.use_RoPE)
        
        self.feed_forward = FeedForward(
            hidden_size=args.hidden_size,
            hidden_dim=4 * args.hidden_size,
            multiple_of=args.multiple_of
        )
        
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.hidden_size, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        
        """
        TransformerBlock 的 FFN 操作.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """

        h = self.attention(self.attention_norm(x), mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
    

class Transformer(nn.Module):
    
    """
        Transformer (llama style).

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """
    
    def __init__(
        self, 
        args: ModelArgs
    ) : 

        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = nn.Linear(args.vocab_size, args.hidden_size)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))

        self.norm = RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.output = nn.Linear(args.hidden_size, args.vocab_size)

    def forward(self, tokens: torch.Tensor) : 
        _bsz, seq_length, vocab_size = tokens.shape
        h = self.tok_embeddings(tokens)
        mask = None
        
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
    

if __name__ == "__main__" : 
    
    model_args = ModelArgs(
        hidden_size = 2048,
        n_layers = 16,
        n_heads = 16,
        vocab_size = 9144,  # defined later by tokenizer
        multiple_of = 256,  # make SwiGLU hidden layer size multiple of large power of 2
        norm_eps = 1e-5,
        use_RMS = True,
        use_RoPE = True,
        max_batch_size = 32,
        max_seq_len = 2048,
    )
    
    transformer = Transformer(model_args).cuda()
    print(transformer)
    