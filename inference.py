import torchaudio
import torch
from hw_hifi.model import Generator
from hw_hifi.collate_fn.mel_spectrogram import CustomMelSpectrogram


device = torch.device('cuda:0')

checkpoint = torch.load('final_data/model.pth')

aa = checkpoint['config']['generator_model']

model = Generator(**aa['args'])
model.load_state_dict(checkpoint['state_dict_gen'])
model = model.to(device)
model.remove_weight_norm()
model.eval()
for k in range(1, 6):
    aud, sr = torchaudio.load(f'test_data/audio_{k}.wav')
    sfft = CustomMelSpectrogram({})
    mel_spec = sfft(aud).to(device)

    gen_aud = model(mel_spectrogram=mel_spec)
    gen_aud = gen_aud['predicted_audios']

    torchaudio.save(f'test_data/audio_test_{k}.wav', gen_aud.squeeze(1).cpu(), sr)


