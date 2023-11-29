import logging
import torch
import torch.nn as nn
from typing import List
from .mel_spectrogram import CustomMelSpectrogram

logger = logging.getLogger(__name__)


def adder(vec, v):
    if vec is None:
        vec = v
    else:
        size_1, size_2 = vec.shape[-1], v.shape[-1]
        pad = size_1 - size_2
        vec = nn.functional.pad(vec, (0, max(-pad, 0)))
        v = nn.functional.pad(v, (0, max(pad, 0)))
        vec = torch.cat([vec, v])
    return vec


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    audio = None

    for item in dataset_items:
        audio = adder(audio, item['audio'])
    specs_maker = CustomMelSpectrogram(dataset_items[0]['spec_cfg'])
    mel_spec = specs_maker(audio)
    result_batch = {'audios': audio,
                    'mel_spectrogram': mel_spec}
    return result_batch

