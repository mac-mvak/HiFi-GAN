import torch.nn as nn
import numpy as np
from .blocks import MRF, PeriodDiscriminator, ScaleDiscriminator, MPD, MSD
from torch.nn.utils import weight_norm, remove_weight_norm



class Generator(nn.Module):
    def __init__(self, h_u, k_u, k_r, D_rs, l_relu_slope, **kwargs):
        super().__init__()
        k_u = np.array(k_u, dtype=int)
        k_r = np.array(k_r, dtype=int)
        D_rs = np.array(D_rs, dtype=int)
        self.blocks_seq = nn.Sequential()
        self.blocks_seq.append(
            nn.Conv1d(80, h_u, kernel_size=7, padding='same'))
        self.forw_remove, self.meth_remove = [0], []
        cur_channels = h_u
        for l in range(k_u.shape[0]):
            self.blocks_seq.append(nn.LeakyReLU(l_relu_slope))
            ker_size = k_u[l]
            p_size = ker_size // 2
            self.blocks_seq.append(
                nn.ConvTranspose1d(cur_channels, cur_channels//2, kernel_size=ker_size,
                                    stride=p_size, padding=(ker_size-p_size)//2)
            )
            #self.forw_remove.append(len(self.blocks_seq) - 1)
            cur_channels //= 2
            self.blocks_seq.append(
                MRF(cur_channels, k_r, D_rs, l_relu_slope)
            )
            self.meth_remove.append(len(self.blocks_seq) - 1)
        self.blocks_seq.append(
            nn.LeakyReLU(l_relu_slope)
        )
        self.blocks_seq.append(
            nn.Conv1d(cur_channels, 1, 7, padding='same')
        )
        #self.forw_remove.append(len(self.blocks_seq) - 1)
        self.blocks_seq.append(
            nn.Tanh()
        )

    def forward(self, mel_spectrogram, **kwargs):
        out = self.blocks_seq(mel_spectrogram)
        return {'predicted_audios': out}
             
    def remove_weight_norm(self):
        #for i in self.forw_remove:
        #    remove_weight_norm(self.blocks_seq[i])
        for j in self.meth_remove:
            self.blocks_seq[j].remove_weight_norm()


class Discriminator(nn.Module):
    def __init__(self, k_msd, periods_mpd, l_relu_slope, **kwargs):
        super().__init__()
        self.msd = MSD(k_msd, l_relu_slope)
        self.mpd = MPD(periods_mpd, l_relu_slope)
    
    def forward(self, audios, predicted_audios, **kwargs):
        y_ds_true, y_ds_pred, layer_true, layer_pred = self.msd(audios, predicted_audios)
        y_ds_true1, y_ds_pred1, layer_true1, layer_pred1 = self.mpd(audios, predicted_audios)
        y_ds_true = y_ds_true + y_ds_true1
        y_ds_pred = y_ds_pred + y_ds_pred1
        layer_true = layer_true + layer_true1
        layer_pred = layer_pred + layer_pred1
        return {
            'true_disc': y_ds_true,
            'generated_disc': y_ds_pred,
            'f_map_real': layer_true,
            'f_map_generated': layer_pred
        }
