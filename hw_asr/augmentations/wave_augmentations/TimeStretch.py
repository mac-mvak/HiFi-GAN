import random
import torch
import librosa
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.rate = kwargs['fixed_rate']

    def __call__(self, data: Tensor):
        ans = librosa.effects.time_stretch(data.squeeze().numpy(), rate=self.rate)
        return torch.from_numpy(ans).unsqueeze(0)


class RandomTimeStretch(TimeStretch):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p

    def __call__(self, data:Tensor):
        if random.random() < self.p:
            return super().__call__(data)
        else:
            return data

