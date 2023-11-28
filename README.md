# ASR project barebones

## Installation guide

Copy this repo.

```shell
pip install -r ./requirements.txt
gdown https://drive.google.com/uc?id=1ym2rlH_CUVZggy81rIJVytBjZrAVE1X3 -O final_data/model.pth
```

Then run scripts using `train.py` and `test.py`. `test.py` will print metrics in `output_metrics.json`.

Now we will define scripts.

1. `config_test_clean.json` -- get results for Librispeech-clean.
2. `config_test_other.json` -- get results for Librispeech-other.
3. `config_train_first.json` -- train on  Librispeech-clean.
4. `config_train_finetune.json` -- finetune on  Librispeech-other.


## Wandb Report

[Link to report](https://wandb.ai/svak/asr_project/reports/ASR-project---Vmlldzo1Nzg2Mzkx)

