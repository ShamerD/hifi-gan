# Structure
* `logger/` defines logging logic (WandB for now)
* `loss/` defines FastSpeech loss
* `model/` contains definition of FastSpeech model as well as other models (aligner and vocoder) used for training and inference
* `scheduler/` defines learning rate scheduler(s)
* `trainer/` defines Trainer - main training logic
* `utils/` and `utils/data/` contains utility functions
* `config.json` is a default config which will be used if none provided