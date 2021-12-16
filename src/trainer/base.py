from abc import abstractmethod
from typing import Optional

import torch
from numpy import inf

from src.logger import get_visualizer
from src.utils.config_parser import ConfigParser


class BaseGANTrainer:
    """
    Base class for all trainers
    """

    def __init__(self,
                 generator: torch.nn.Module,
                 gen_criterion: torch.nn.Module,
                 gen_optimizer: torch.optim.Optimizer,
                 config: ConfigParser,
                 device: torch.device,
                 discriminator: Optional[torch.nn.Module] = None,
                 disc_criterion: Optional[torch.nn.Module] = None,
                 disc_optimizer: Optional[torch.optim.Optimizer] = None,
                 gen_scheduler: Optional = None,
                 disc_scheduler: Optional = None):
        self.device = device
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.generator = generator
        self.gen_criterion = gen_criterion
        self.gen_optimizer = gen_optimizer
        self.gen_scheduler = gen_scheduler

        self.discriminator = discriminator
        self.disc_criterion = disc_criterion
        self.disc_optimizer = disc_optimizer
        self.disc_scheduler = disc_scheduler

        # for interrupt saving
        self._last_epoch = 0

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = get_visualizer(
            config, self.logger, cfg_trainer["visualize"]
        )

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best)\
                               or (self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        :param only_best: if True, save only best
        """
        gen_arch = type(self.generator).__name__
        disc_arch = type(self.discriminator).__name__ if self.discriminator is not None else None
        state = {
            "gen_arch": gen_arch,
            "epoch": epoch,
            "generator": self.generator.state_dict(),
            "gen_optimizer": self.gen_optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        if self.gen_scheduler is not None:
            state['gen_scheduler'] = self.gen_scheduler.state_dict()

        if disc_arch is not None:
            state.update({
                "disc_arch": disc_arch,
                "discriminator": self.discriminator.state_dict(),
                "disc_optimizer": self.disc_optimizer.state_dict()
            })
            if self.disc_scheduler is not None:
                state['disc_scheduler'] = None

        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not (only_best and save_best):
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["model_config"] != self.config["model_config"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that of "
                "checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.generator.load_state_dict(checkpoint["generator"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
                checkpoint["config"]["gen_optimizer"] != self.config["gen_optimizer"] or
                checkpoint["config"]["gen_scheduler"] != self.config["gen_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])

            if self.gen_scheduler is not None and 'gen_scheduler' in checkpoint:
                self.gen_scheduler.load_state_dict(checkpoint["gen_scheduler"])

        # load discriminator if it is present
        if self.discriminator is not None and 'discriminator' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])

            if self.disc_scheduler is not None and 'disc_scheduler' in checkpoint:
                self.disc_scheduler.load_state_dict(checkpoint['disc_scheduler'])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
