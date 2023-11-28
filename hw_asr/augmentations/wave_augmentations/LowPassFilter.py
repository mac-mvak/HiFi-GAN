import random
import torchaudio
import torch_audiomentations

from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class LowPassFilter(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.filter = torch_audiomentations.LowPassFilter(**kwargs)

    def __call__(self, data: Tensor):
        return self.filter(data.unsqueeze(1)).squeeze(1)
