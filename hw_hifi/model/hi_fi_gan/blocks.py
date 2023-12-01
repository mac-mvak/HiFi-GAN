import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm



class ConvBlock(nn.Module):
    def __init__(self, channels, k, Ds, lrelu_slope):
        super().__init__()
        self.convs = nn.Sequential()
        for l in range(Ds.shape[0]):
            self.convs.append(
                nn.LeakyReLU(negative_slope=lrelu_slope)
            )
            self.convs.append(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=k, 
                          stride=1, dilation=Ds[l], padding='same'))
            )
    
    def forward(self, x):
        return self.convs(x)
    
    def remove_weight_norm(self):
        for i, layer in enumerate(self.convs):
            if i % 2 == 1:
                remove_weight_norm(layer)
    

class ResBlock(nn.Module):
    def __init__(self, channels, k, Drs, lrelu_slope):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        for i in range(Drs.shape[0]):
            self.conv_blocks.append(
                ConvBlock(channels, k, Drs[i, :], lrelu_slope)
            )
    
    def forward(self, x):
        out = 0
        for block in self.conv_blocks:
            conv_out = block(x)
            out = conv_out + out
        return out
    
    def remove_weight_norm(self):
        for layer in self.conv_blocks:
            layer.remove_weight_norm()

class MRF(nn.Module):
    def __init__(self, channels, k_r, D_rs, lrelu_slope):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for l in range(k_r.shape[0]):
            self.res_blocks.append(
                ResBlock(channels, k_r[l], D_rs[l], lrelu_slope)
            )
        self.n_c = k_r.shape[0]

    def forward(self, x):
        out = None
        for module in self.res_blocks:
            res_out = module(x)
            if out is None:
                out = res_out
            else:
                out = out + res_out
        #out = out/self.n_c
        return out
    
    def remove_weight_norm(self):
        for layer in self.res_blocks:
            layer.remove_weight_norm()
    

class PeriodDiscriminator(nn.Module):
    def __init__(self, period, lrelu_slope=0.1, use_spectral_norm = False):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 2**5, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(2**5, 2**7, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(2**7, 2**9, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(2**9, 2**10, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(2**10, 2**10, (5, 1), 1, padding=(2, 0))
        ]
        )
        self.final_conv = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
        self.lrelu = nn.LeakyReLU(lrelu_slope)

    def forward(self, x):
        layers = []

        b, c, t = x.shape
        if (t % self.period) != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), mode='reflect')
        x = x.view(b, c, -1, self.period)

        out = x
        for conv in self.convs:
            out = conv(out)
            out = self.lrelu(out)
            layers.append(out)
        out = self.final_conv(out)
        layers.append(out)
        out = out.reshape(out.shape[0], -1)
        return out, layers


class ScaleDiscriminator(nn.Module):
    def __init__(self, lrelu_slope=0.1, use_spectral_norm = False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 128, 15, 1, padding=7),
            nn.Conv1d(128, 128, 41, 2, groups=4, padding=20),
            nn.Conv1d(128, 256, 41, 2, groups=16, padding=20),
            nn.Conv1d(256, 512, 41, 4, groups=16, padding=20),
            nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 5, 1, padding=2),
        ]
        )
        self.final_conv = nn.Conv1d(1024, 1, 3, 1, padding=1)
        self.lrelu = nn.LeakyReLU(lrelu_slope)

    def forward(self, x):
        layers = []
        out = x
        for conv in self.convs:
            out = conv(out)
            out = self.lrelu(out)
            layers.append(out)
        out = self.final_conv(out)
        layers.append(out)
        out = out.reshape(out.shape[0], -1)
        return out, layers
    

class MPD(nn.Module):
    def __init__(self, periods, l_relu_slope) -> None:
        super().__init__()
        self.period_discrs = nn.ModuleList(
            [PeriodDiscriminator(period, lrelu_slope=l_relu_slope) for period in periods]
        )

    def forward(self, y_true, y_pred):
        y_ds_true, y_ds_pred = [], []
        layers_true, layers_pred = [], []
        for disc in self.period_discrs:
            y_d_true, layer_true = disc(y_true)
            y_d_pred, layer_pred = disc(y_pred)
            y_ds_true.append(y_d_true)
            y_ds_pred.append(y_d_pred)
            layers_true = layers_true + layer_true
            layers_pred = layers_pred + layer_pred
        return y_ds_true, y_ds_pred, layer_true, layer_pred

class MSD(nn.Module):
    def __init__(self, k, l_relu_slope):
        super().__init__()
        self.scale_discrs = nn.ModuleList(
            [ScaleDiscriminator(lrelu_slope=l_relu_slope, use_spectral_norm=True)]
        )
        self.poolings = nn.ModuleList()
        for _ in range(k-1):
            self.scale_discrs.append(
                ScaleDiscriminator(lrelu_slope=l_relu_slope)
            )
            self.poolings.append(nn.AvgPool1d(4, stride=2, padding=2))

    def forward(self, y_true, y_pred):
        y_ds_true, y_ds_pred = [], []
        layers_true, layers_pred = [], []
        for i, disc in enumerate(self.scale_discrs):
            if i!=0:
                y_true, y_pred = self.poolings[i-1](y_true), self.poolings[i-1](y_pred)
            y_d_true, layer_true = disc(y_true)
            y_d_pred, layer_pred = disc(y_pred)
            y_ds_true.append(y_d_true)
            y_ds_pred.append(y_d_pred)
            layers_true = layers_true + layer_true
            layers_pred = layers_pred + layer_pred
        return y_ds_true, y_ds_pred, layer_true, layer_pred
