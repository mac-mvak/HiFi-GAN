from torch import nn
from torch.nn import Sequential

#from hw_asr.base import BaseModel


class JasperSubSubBlock(nn.Module):
    def __init__(self, kernel, output_channels, p_dropout):
        self.net = nn.Sequential(
            nn.Conv1d
        )


class JasperSubBlock:
    def __init__(self, r, kernel, output_channels, p_dropout):
        self.r = r
        self.kernel = kernel,
        self