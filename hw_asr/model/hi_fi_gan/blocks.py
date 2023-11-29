import torch
import torch.nn as nn
import numpy as np



class ConvBlock(nn.Module):
    def __init__(self, channels, k, Ds, lrelu_slope):
        super().__init__()
        self.convs = nn.Sequential()
        for l in range(Ds.shape[0]):
            self.convs.append(
                nn.LeakyReLU(negative_slope=lrelu_slope)
            )
            self.convs.append(
                nn.Conv1d(channels, channels, kernel_size=k, 
                          stride=1, dilation=Ds[l], padding='same')
            )
    
    def forward(self, x):
        return self.convs(x)
    

class ResBlock(nn.Module):
    def __init__(self, channels, k, Drs, lrelu_slope):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        for i in range(Drs.shape[0]):
            self.conv_blocks.append(
                ConvBlock(channels, k, Drs[i, :], lrelu_slope)
            )
    
    def forward(self, x):
        out = x
        for block in self.conv_blocks:
            conv_out = block(out)
            out = conv_out + out
        return out

class MRF(nn.Module):
    def __init__(self, channels, k_r, D_rs, lrelu_slope):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for l in range(k_r.shape[0]):
            self.res_blocks.append(
                ResBlock(channels, k_r[l], D_rs[l], lrelu_slope)
            )

    def forward(self, x):
        out = None
        for module in self.res_blocks:
            res_out = module(x)
            if out is None:
                out = res_out
            else:
                out = out + res_out
        return out

class Generator(nn.Module):
    def __init__(self, h_u, k_u, k_r, D_rs, l_relu_slope):
        super().__init__()
        k_u = np.array(k_u, dtype=int)
        k_r = np.array(k_r, dtype=int)
        D_rs = np.array(D_rs, dtype=int)
        self.blocks_seq = nn.Sequential()
        self.blocks_seq.append(
            nn.Conv1d(80, h_u, kernel_size=7, padding='same')
        )
        cur_channels = h_u
        for l in range(k_u.shape[0]):
            self.blocks_seq.append(nn.LeakyReLU(l_relu_slope))
            ker_size = k_u[l]
            p_size = ker_size // 2
            self.blocks_seq.append(
                nn.ConvTranspose1d(cur_channels, cur_channels//2, kernel_size=ker_size,
                                    stride=p_size, padding=(ker_size-p_size)//2)
            )
            cur_channels //= 2
            self.blocks_seq.append(
                MRF(cur_channels, k_r, D_rs, l_relu_slope)
            )
        self.blocks_seq.append(
            nn.LeakyReLU(l_relu_slope)
        )
        self.blocks_seq.append(
            nn.Conv1d(cur_channels, 1, 7, padding='same')
        )
        self.blocks_seq.append(
            nn.Tanh()
        )

    def forward(self, x):
        out = x
        for bl in self.blocks_seq:
            new_out = bl(out)
            out = new_out
        return out
             






