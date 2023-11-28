from torch import nn
from hw_asr.model.utils import _conv_shape_transform1d, _conv_shape_transform2d
import torch

from hw_asr.base import BaseModel


class DeepSpeech(BaseModel):
    def __init__(self, n_feats, n_class, conv_type, convs_params, grus_params,
                **batch):
        super().__init__(n_feats, n_class, **batch)
        self.conv_type = conv_type
        self.convs_params = convs_params
        self.grus_len = len(grus_params)
        self.convs = nn.Sequential()
        size_after_feat = n_feats
        for conv_params in self.convs_params:
            if self.conv_type == "Conv1d":
                self.convs.append(nn.Conv1d(**conv_params['convolution']))
                self.convs.append(nn.BatchNorm1d(**conv_params['batch_norm']))
                size_after_feat = conv_params['convolution']["out_channels"]
            elif self.conv_type == "Conv2d":
                self.convs.append(nn.Conv2d(**conv_params['convolution']))
                self.convs.append(nn.BatchNorm2d(**conv_params['batch_norm']))
                size_after_feat = _conv_shape_transform2d(size_after_feat, dim=0, **conv_params['convolution'])
            self.convs.append(nn.ReLU())
        self.grus = nn.ModuleList()
        self.bnorms = nn.ModuleList()
        if self.conv_type == "Conv2d":
            size_after_feat *= self.convs_params[-1]['convolution']["out_channels"]
        for i, gru_params in enumerate(grus_params):
            if i != 0:
                self.grus.append(nn.GRU(**gru_params['gru']))
            else:
                self.grus.append(nn.GRU(input_size = size_after_feat, **gru_params['gru']))
            self.bnorms.append(nn.BatchNorm1d(**gru_params['batch_norm']))
        self.fc = nn.Linear(in_features=2*grus_params[-1]['gru']['hidden_size'], 
                            out_features=n_class)


    def forward(self, spectrogram, **batch):
        if self.conv_type == "Conv2d":
            spectrogram = spectrogram.unsqueeze(1)
        conv_spec = self.convs(spectrogram)
        out = conv_spec
        if self.conv_type == "Conv2d":
            out = torch.permute(conv_spec, (0, 3, 1, 2))
            out = out.reshape(*out.shape[:2], -1).transpose(1, 2)
        h = None
        for i, tup in enumerate(zip(self.grus, self.bnorms)):
            gru, bnorm = tup
            out, h = gru(out.transpose(1, 2), h)
            out = bnorm(out.transpose(1, 2))
            if i != self.grus_len - 1:
                out = nn.functional.relu(out)
        return {"logits": self.fc(out.transpose(1, 2))}

    def transform_input_lengths(self, input_lengths):
        for conv_params in self.convs_params:
            if self.conv_type == "Conv2d":
                input_lengths = _conv_shape_transform2d(input_lengths, **conv_params['convolution'], 
                                                      dim=1)
            else:
                input_lengths = _conv_shape_transform1d(input_lengths, **conv_params['convolution'])
        return input_lengths 



