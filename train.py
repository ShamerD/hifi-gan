import argparse
import dataclasses
import warnings

import torch

import src.loss as module_loss
import src.model as module_model
import src.utils.data as module_data
from src.trainer import GANTrainer
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

    # build models, then print them
    wav2mel = module_data.MelSpectrogram(featurizer_config)
    generator = module_model.HiFiGenerator(model_config)
    use_discriminator = config['trainer'].get("use_discriminator", True)

    logger.info("Generator:")
    logger.info(generator)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    generator = generator.to(device)
    wav2mel = wav2mel.to(device)

    if len(device_ids) > 1:
        generator = torch.nn.DataParallel(generator, device_ids=device_ids)

    # get function handles of loss and metrics
    gen_loss = config.init_obj(config["losses"]["generator"],
                               module_loss, use_discriminator=use_discriminator).to(device)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, generator.parameters())
    gen_optimizer = config.init_obj(config["optimizers"]["generator"], torch.optim, trainable_params)
    gen_scheduler = None
    if "lr_schedulers" in config.config and "generator" in config["lr_schedulers"]:
        gen_scheduler = config.init_obj(config["lr_schedulers"]["generator"],
                                        torch.optim.lr_scheduler, gen_optimizer)

    # repeat everything for discriminator if needed
    discriminator = None
    disc_loss = None
    disc_optimizer = None
    disc_scheduler = None
    if use_discriminator:
        discriminator = module_model.HiFiDiscriminator(model_config)
        logger.info("Discriminator:")
        logger.info(discriminator)
        discriminator = discriminator.to(device)

        if len(device_ids) > 1:
            discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids)

        disc_loss = config.init_obj(config["losses"]["discriminator"], module_loss).to(device)
        trainable_params = filter(lambda p: p.requires_grad, discriminator.parameters())
        disc_optimizer = config.init_obj(config["optimizers"]["discriminator"], torch.optim, trainable_params)
        if "lr_schedulers" in config.config and "discriminator" in config["lr_schedulers"]:
            disc_scheduler = config.init_obj(config["lr_schedulers"]["discriminator"],
                                             torch.optim.lr_scheduler, disc_optimizer)

    trainer = GANTrainer(
        generator,
        gen_loss,
        gen_optimizer,
        wav2mel,
        config=config,
        device=device,
        data_loader=dataloaders["train"],
        valid_data_loader=dataloaders["val"],
        inference_data_loader=dataloaders["inference"],
        discriminator=discriminator,
        disc_criterion=disc_loss,
        disc_optimizer=disc_optimizer,
        gen_scheduler=gen_scheduler,
        disc_scheduler=disc_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None),
        log_step=config["trainer"].get("log_step", None)
    )

    trainer.train()


if __name__ == "__main__":
    fix_seed()
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
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizers;generator;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
