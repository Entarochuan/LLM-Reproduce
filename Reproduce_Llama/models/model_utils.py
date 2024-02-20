"""
    model_utils
    模型所需组件实现
        RMSNorm, RoPE
"""

from torch import nn 
import torch
from einops import rearrange, repeat

class RMSNorm(nn.Module):
    
    """ RMS Mormalization

        Args:
            dim : hidden size
            eps : epsilon
    
    """    
    
    def __init__(
        self, 
        dim:int, 
        eps:float=1e-5
    ) -> None: 
        super().__init__()
        
        self.eps = eps 
        self.weights = torch.ones(dim)
    
    def forward(self, x:torch.Tensor):
        """ 
        Args:
            x : Tensor
        Returns:
           output =  weight * x / RMS(x)
    
        """
        
        variance = x.pow(2).to(torch.float32).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weights.dtype in [torch.float16, torch.bfloat16]:
            x = x.to(self.weights.dtype)

        return self.weights * x 
    

class Rotary_Positional_Embeedding(nn.Module):
    """RoPE : Rotary_Positional_Embeedding (Some changes may be made, so encapsulated as a class)
    
        Reference:
            1. https://zhuanlan.zhihu.com/p/630082091 
            2. https://zhuanlan.zhihu.com/p/642884818
            
            V1.0 实现参考 : Flash Attention -> 按照公式版本的实现

        Args:
            hidden_size(int) 
            base(int) : θn基座
    """
    
    def __init__(
        self,
        hidden_size:int, 
        base:float=10000.0
    ) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size
        self.base = base

    def rotate_half(self, x, interleaved:bool=False):
        
        """ Rotate x 
        
            interleaved(bool) : 是否需要进行置换的操作
        
        """
        if not interleaved:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        else:
            # x.shape = (bsz, seq_len, dim)
            x1, x2 = x[..., ::2], x[..., 1::2] # 奇数负，偶数正 
            x_stack = torch.stack((-x2, x1), dim=-1) # (-q1, q0, -q3, q2, ...)
            return rearrange(x_stack, '... d two -> ... (d two)', two=2)

    def apply_rotary_emb_torch(
        self, 
        x:torch.Tensor, 
        rotate_dim:int, 
        interleaved:bool=False
    ) -> torch.Tensor:
        
        """ 参考了Flash Attention实现的Rotary Embedding 
            支持 multi-head 形式输入。
            
            Args: 
                x(torch.Tensor) : 需要应用RoPE的tensor
                rotate_dim(int) : 需要应用的长度
                
            freq_k(n) = k / base ^ (2i/dim) 
            
            后续的改进策略 : 是否能够添加一部分的可学习参数?
             
        """
        
        # x.shape : (bs, seq_len, dim) 
        # print(x.shape)
        
        if len(x.shape) == 3 : 
            bsz, seq_len, hidden_size = x.shape
        elif len(x.shape) == 4 : 
            bsz, seq_len, head_num, hidden_size = x.shape
            
        rotate_dim = min(rotate_dim, hidden_size) // 2
        freqs = torch.empty(size=(seq_len, rotate_dim)) # (0, 1, 2, ..., ...) 
        rotate_freqs = 1.0 / (10000.0 ** (torch.arange(rotate_dim) / rotate_dim))
        # print(rotate_freqs)
        for i in range(seq_len):
            freqs[i] = i * rotate_freqs
        
        cos, sin = torch.empty(size=(seq_len, rotate_dim*2)), torch.empty(size=(seq_len, rotate_dim*2))
        
        cos[:, ::2], cos[:, 1::2] = torch.cos(freqs), torch.cos(freqs)
        sin[:, ::2], sin[:, 1::2] = torch.sin(freqs), torch.sin(freqs)
        
        rotate_x = self.rotate_half(x, interleaved=interleaved) # cos * x + sin * rotate_x
        # print(rotate_x[0][0])
        if len(x.shape) == 4 : 
            cos = cos.unsqueeze(1).repeat(1, 1, head_num, 1)
            sin = sin.unsqueeze(1).repeat(1, 1, head_num, 1)
            
        return cos * x + sin * rotate_x

        
def test() : 
    
    # RMS Norm Test
    # RMS_test = RMSNorm(dim=128, eps=1e-5)
    # x = torch.arange(1, 129) 
    # x = x.unsqueeze(0).unsqueeze(0)
    # x = x.expand(16, 32, 128)
    # print(x.shape)
    # x1 = RMS_test(x)
    # print(x1[0, 0])
    
    # Rotary Embedding Test
    Rotary_test = Rotary_Positional_Embeedding(hidden_size=128)
    x = torch.arange(1, 129)
    # x = torch.ones_like()
    x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x = x.expand(16, 32, 4, 128)
    # print(x[0, 0])
    # print(Rotary_test.rotate_half(x, True)[0, 0])
    # freqs = Rotary_test.apply_rotary_emb_torch(x, 128, True)
    # print(freqs[2])
    rotary_emb_x = Rotary_test.apply_rotary_emb_torch(x, 128, True)
    print(rotary_emb_x[0][4]) # 1/16 实现完成，明天测试。

if __name__ == "__main__"  :
    
   test()

    
    

