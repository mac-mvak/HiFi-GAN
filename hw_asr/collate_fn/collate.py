import logging
import torch
import torch.nn as nn
from typing import List

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
    audio = []
    spectrogram = None
    text_encoded = None
    spectrogram_length = []
    text_encoded_length = []
    duration = []
    audio_paths = []
    text = []

    for item in dataset_items:
        audio.append(item['audio'])
        spectrogram = adder(spectrogram, item['spectrogram'])
        text_encoded = adder(text_encoded, item['text_encoded'])
        text_encoded_length.append(item['text_encoded'].shape[-1])
        duration.append(item['duration'])
        audio_paths.append(item['audio_path'])
        text.append(item['text'])
        spectrogram_length.append(item['spectrogram'].shape[-1])
    result_batch = {'spectrogram' : spectrogram, 
                    'spectrogram_length' : torch.tensor(spectrogram_length, dtype=int),
                    'text_encoded' : text_encoded,
                    'text_encoded_length': torch.tensor(text_encoded_length, dtype=int),
                    'duraition' : duration,
                    'audio_path' : audio_paths,
                    'text': text,
                    'audio': audio}
    return result_batch

