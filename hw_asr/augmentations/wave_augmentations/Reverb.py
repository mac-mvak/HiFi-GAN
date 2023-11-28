import random
import torchaudio_augmentations
import torch
import torch.nn.functional as F
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class Reverb(AugmentationBase):
    def __init__(self, p=1.0, *args, **kwargs):
        self.reverb = torchaudio_augmentations.Reverb(**kwargs)
        self.p = p

    def __call__(self, data: Tensor):
        if random.random() < self.p:
            return self.reverb(data)
        else:
            return data
