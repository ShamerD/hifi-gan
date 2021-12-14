import argparse
import dataclasses
import sys
import warnings

import torch
from torch.optim.lr_scheduler import LambdaLR

import src.loss as module_loss
import src.model as module_model
import src.utils.data as module_data
from src.scheduler import LinearWarmupScheduler
from src.trainer import Trainer
from src.utils import prepare_device, fix_seed
from src.utils.config_parser import ConfigParser, CustomArgs
from src.utils.data import get_dataloaders

warnings.filterwarnings("ignore", category=UserWarning)


def main(config: ConfigParser):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config['data'])

    # update featurizer and model config
    featurizer_config = config.init_obj(config['mel_config'], module_data)
    config['mel_config']['args'].update(dataclasses.asdict(featurizer_config))

    model_config = config.init_obj(config['model_config'], module_model)
    config['model_config']['args'].update(dataclasses.asdict(model_config))

    # build all models, then print FastSpeech to console
    wav2mel = module_data.MelSpectrogram(featurizer_config)
    model = module_model.FastSpeech(model_config)
    aligner = module_model.GraphemeAligner(featurizer_config)
    vocoder = module_model.Waveglow()

    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    wav2mel = wav2mel.to(device)
    aligner = aligner.to(device)
    vocoder = vocoder.to(device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss = config.init_obj(config["loss"], module_loss).to(device)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    if config["lr_scheduler"]["type"] == "LinearWarmupScheduler":
        # Don't know how to configure this out
        len_epoch = config["trainer"].get("len_epoch", len(dataloaders["train"]))
        total_steps = config["trainer"]["epochs"] * len_epoch
        warmup_steps = config["lr_scheduler"]["args"]["warmup_steps"]
        if type(warmup_steps) == float:
            warmup_steps = int(warmup_steps * total_steps)
        lr_scheduler = LambdaLR(optimizer,
                                lr_lambda=LinearWarmupScheduler(
                                    model_config.d_model, total_steps=total_steps, warmup_steps=warmup_steps)
                                )
    else:
        lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        wav2mel,
        aligner,
        vocoder,
        loss,
        optimizer,
        config=config,
        device=device,
        data_loader=dataloaders["train"],
        valid_data_loader=dataloaders["val"],
        inference_data_loader=dataloaders["inference"],
        lr_scheduler=lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None),
        log_step=config["trainer"].get("log_step", None)
    )

    trainer.train()


if __name__ == "__main__":
    fix_seed()
    sys.path.append('waveglow/')
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
