"""
    multi_head_attention
    自注意力实现v1.0
"""

import torch 
from torch import nn
from einops import rearrange
import torch.nn.functional as F

from model_utils import RMSNorm

class Multi_Head_Self_Attention(nn.Module) : 
    
    """ Multi-Head-Attention

        Args:
        
            hidden_size(int) : size of qkv
            num_heads(int)   : head num
            dropout(float)   : dropout
            eps(float)       : epsilon 
            use_RMS(bool)    : whether to use RMSNorm
            
        Returns:

            x : tensor(batch_size, len, hidden_size)
            attn : attention score
    """
    
    def __init__(
        self,
        hidden_size:int, 
        num_heads:int, 
        dropout:float, 
        eps:float=1e-6, 
        use_RMS:bool=True
    ) -> None: 
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = hidden_size // num_heads
        
        self.Wqkv = nn.Linear(hidden_size, 3*hidden_size)
        self.attention = ScaledDotProductAttention(temperature=hidden_size ** 0.5)
        self.Wo = nn.Linear(hidden_size, hidden_size)
        
        if use_RMS : 
            self.norm = RMSNorm(dim=hidden_size, eps=eps)
        else :
            self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x, attention_mask=None):
        
        # qkv : (batch_size, seq_len, 3, num_head, head_dim)
        
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads, d=self.head_dim) # resize qkv 
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # (b, s, h, d) -> (b, h, s, d)
    
        output, attn = self.attention(q, k, v, attention_mask)
        
        output = rearrange(output, 'b h s d -> b s (h d)')
        x += output
        x = self.norm(x)
        x = self.Wo(x)
        
        return x, attn
        
        
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
    

if __name__ == "__main__" : 
    
    MHA_test = Multi_Head_Self_Attention(hidden_size=128, num_heads=4, dropout=0.5, eps=1e-5)
    print(MHA_test)
    x = torch.randn(32, 64, 128)
    x_o, attn = MHA_test(x)
    print(x_o.shape)
    
   