import random
import torchaudio
import librosa
import random
import torch


from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class AddNoise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        
        filename = librosa.ex('pistachio')
        self.noise, _ = torchaudio.load(filename)
        self.snr = Tensor([kwargs['snr']])
        self.p = kwargs['p']

    def transform(self, speech, noise, snr):
        speech, noise = speech[:,:noise.shape[1]], noise[:, :speech.shape[1]]
        speech_rms = speech.norm(p=2)
        noise_rms = noise.norm(p=2)
        snr = 10 ** (-snr / 20)
        scale = snr * speech_rms / noise_rms 
        return torch.clamp(speech + scale * noise, -1, 1)


    def __call__(self, data: Tensor):
        if random.random() < self.p:
            return self.transform(data, self.noise, self.snr)
            
        else:
            return data

