# %%
import torch
import torch.nn as nn

class CasualConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size, padding=kernel_size-1)
        self.kernel_size = kernel_size
    
    def forward(self, x):
        return self.conv(x)[:, :, :-self.kernel_size+1]

class EncodingLayer(nn.Module):
    def __init__(self, d_model, d_proj, kernel1, kernel2) -> None:
        super().__init__()
        self.conv1 = CasualConv(d_model, d_proj, kernel1)
        self.conv2 = CasualConv(d_proj, d_model, kernel2)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm((d_model,))
    
    def forward(self, seq):
        x = seq.transpose(-1, -2)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x.transpose(-1, -2)
        x = self.norm(x)
        return seq + x


