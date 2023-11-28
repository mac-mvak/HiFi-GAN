import random
import torchaudio
import torch_audiomentations

from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class PitchShift(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.pitch_shift = torch_audiomentations.PitchShift(**kwargs)

    def __call__(self, data: Tensor):
        return self.pitch_shift(data)
