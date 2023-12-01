import torch.nn as nn
from hw_hifi.collate_fn.collate import CustomMelSpectrogram



class MelLoss(nn.Module):
    def __init__(self, spec_config, **kwargs):
        super().__init__()
        self.mel_spec = CustomMelSpectrogram(spec_config)
        self.loss = nn.L1Loss()

    def forward(self, mel_spectrogram, predicted_audios, **kwargs):
        pred_specs = self.mel_spec(predicted_audios.squeeze(1))
        return self.loss(mel_spectrogram, pred_specs)
    

class FMLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.L1Loss()


    def forward(self, f_map_real, f_map_generated, **kwargs):
        loss = 0
        for f_real, f_generated in zip(f_map_real, f_map_generated):
            loss = loss + self.loss(f_real, f_generated)
        return loss

class GANLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, generated_disc, **kwargs):
        loss = 0
        for gen in generated_disc:
            l1 = ((gen-1)**2).mean()
            loss = loss + l1
        return loss
    

class GeneratorLoss(nn.Module):
    def __init__(self, la_fm, la_mel , **cfg):
        super().__init__()
        self.mel_loss = MelLoss(**cfg)
        self.fm_loss = FMLoss(**cfg)
        self.gan_loss = GANLoss(**cfg)
        self.la_fm = la_fm
        self.la_mel = la_mel

    def forward(self, **batch):
        mel_loss = self.mel_loss(**batch)
        fm_loss = self.fm_loss(**batch)
        gan_loss = self.gan_loss(**batch)
        loss =  self.la_mel * mel_loss + self.la_fm * fm_loss +  gan_loss
        return {
            "loss": loss,
            "mel_loss": mel_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "gan_loss": gan_loss.detach()
        }







