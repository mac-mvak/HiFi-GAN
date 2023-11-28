from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class LSTM_Linear(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, num_layers=1, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.lstm = nn.LSTM(
            input_size = n_feats,
            hidden_size=fc_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(2 * fc_hidden, n_class)

    def forward(self, spectrogram, **batch):
        lstm_out, _ = self.lstm(spectrogram.transpose(1, 2))
        return {"logits": self.fc(lstm_out)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here

