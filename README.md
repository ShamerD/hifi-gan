# hifi-gan
[HiFi-GAN](https://arxiv.org/pdf/2010.05646.pdf) is a vocoder in TTS pipeline. 

Report can be found [here](https://wandb.ai/shamerd/nv_project/reports/TTS-Vocoder---VmlldzoxMzU4MzYw).

# Installation
```shell
chmod +x setup.sh && ./setup.sh
```

If something with downloading model goes wrong, you can manually download model weights from [here](https://drive.google.com/file/d/1-h8A3AtB2Z_IQbmv2mROC3z4xew_Kehw/view?usp=sharing)
and put them under `resources/hifi-gan.pth`

# Usage
## Training
```shell
python3 train.py -c <config_file> [-r <resume_path>]
```

## Inference
Inference script takes as input directory with spectrograms in `.npy` format (e.g. `data/default_example_spec`)
and produces audio samples in target directory
```shell
python3 inference.py -c <config_file> -r <checkpoint_file> [-s <source_dir>] [-t <target_dir>]
```

By default audio for 3 default samples will be generated:
```shell
python3 inference.py -c configs/main.json -r resources/hifi-gan.pth
```

Default samples:
* `A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest`
* `Massachusetts Institute of Technology may be best known for its math, science and engineering education`
* `Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space`

## Project structure
* `configs/` contains configs which were used to train model
* `data/` contains data (LJSpeech downloads there by default), trainval split (indices in dataset) and default examples
* `src/` contains source codes
* `train.py` is a training script (it downloads all needed data if it is not present)
* `inference.py` is an inference script which takes .npy spectrograms and outputs audio files in a directory