import os
from speechbrain.utils.data_utils import download_file


def get_model_lower(model_name):
    model_path = f"hw_asr/text_encoder/lm_s/{model_name}.arpa"
    lower_model_path = f"hw_asr/text_encoder/lm_s/lower_{model_name}.arpa"
    if not os.path.exists(lower_model_path):
        with open(model_path, 'r') as f1:
            with open(lower_model_path, "w") as f2:
                for line in f1:
                    f2.write(line.lower())
    return lower_model_path



def get_texts():
    path = "data/datasets/librispeech/texts/librispeech-vocab.txt"
    lower_path = "data/datasets/librispeech/texts/lower_librispeech-vocab.txt"
    if (not os.path.exists(path)) and (not os.path.exists(lower_path)):
        download_file("https://www.openslr.org/resources/11/librispeech-vocab.txt", path)
        with open(path, "r") as f1:
            with open(lower_path, "w") as f2:
                for line in f1:
                    f2.write(line.lower())
    words = []
    with open(lower_path, "r") as f:
        for line in f:
            words.append(line.strip())
    return words






