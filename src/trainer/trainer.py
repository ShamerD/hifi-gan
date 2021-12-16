import random
from typing import Optional

import PIL
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.logger.utils import plot_spectrogram_to_buf
from src.model import HiFiGenerator, HiFiDiscriminator
from src.utils import inf_loop, MetricTracker
from src.utils.config_parser import ConfigParser
from src.utils.data import Batch, MelSpectrogram
from .base import BaseGANTrainer


class GANTrainer(BaseGANTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            generator: HiFiGenerator,
            gen_criterion: nn.Module,
            gen_optimizer: torch.optim.Optimizer,
            wav2mel: MelSpectrogram,
            config: ConfigParser,
            device: torch.device,
            data_loader: torch.utils.data.DataLoader,
            valid_data_loader=None,
            inference_data_loader=None,
            discriminator: Optional[HiFiDiscriminator] = None,
            disc_criterion: Optional[nn.Module] = None,
            disc_optimizer: Optional[torch.optim.Optimizer] = None,
            gen_scheduler=None,
            disc_scheduler=None,
            len_epoch: Optional[int] = None,
            skip_oom: bool = True,
            log_step: Optional[int] = None
    ):
        super().__init__(generator, gen_criterion, gen_optimizer,
                         config, device,
                         discriminator, disc_criterion, disc_optimizer,
                         gen_scheduler, disc_scheduler)
        self.wav2mel = wav2mel

        self.skip_oom = skip_oom
        self.config = config
        self.data_loader = data_loader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        self.inference_loader = inference_data_loader
        self.do_inference = self.inference_loader is not None

        self.use_discriminator = self.discriminator is not None

        self.log_step = log_step if log_step is not None else 100

        metrics_to_log = ["generator_loss"]
        if self.use_discriminator:
            metrics_to_log.extend(["spec_loss", "adv_gen_loss", "feature_loss", "discriminator_loss"])
            metrics_to_log.append("disc_grad_norm")
        metrics_to_log.append("gen_grad_norm")

        self.train_metrics = MetricTracker(
            *metrics_to_log, writer=self.writer
        )

        # pop grad_norms
        metrics_to_log.pop()
        metrics_to_log.pop()
        self.valid_metrics = MetricTracker(
            *metrics_to_log, writer=self.writer
        )

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.generator.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

            if self.use_discriminator:
                clip_grad_norm_(
                    self.discriminator.parameters(), self.config["trainer"]["grad_norm_clip"]
                )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.generator.train()
        if self.use_discriminator:
            self.discriminator.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(
                tqdm(self.data_loader, desc="train", total=self.len_epoch)
        ):
            if batch_idx >= self.len_epoch:
                break

            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.generator.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    if self.use_discriminator:
                        for p in self.discriminator.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            self.train_metrics.update("gen_grad_norm", self.get_grad_norm(self.generator))
            if self.use_discriminator:
                self.train_metrics.update("disc_grad_norm", self.get_grad_norm(self.discriminator))

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {}\n\tGenerator Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch.generator_loss.item()
                    )
                )
                if self.use_discriminator:
                    self.logger.debug(
                        "\tDiscriminator Loss: {:.6f}".format(
                            batch.discriminator_loss.item()
                        )
                    )
                if self.gen_scheduler is not None:
                    self.writer.add_scalar(
                        "gen learning rate", self.gen_scheduler.get_last_lr()[0]
                    )
                if self.disc_scheduler is not None:
                    self.writer.add_scalar(
                        "disc learning rate", self.disc_scheduler.get_last_lr()[0]
                    )

                self._log_predictions(batch)
                self._log_scalars(self.train_metrics)

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.do_inference:
            self._inference_epoch(epoch)

        return log

    def process_batch(
            self,
            batch: Batch,
            is_train: bool,
            metrics: MetricTracker
    ):
        batch = batch.to(self.device)

        # prepare additional data
        with torch.no_grad():
            batch.spec = self.wav2mel(batch.wav)

        batch.wav_gen = self.generator(batch.spec)
        batch.spec_gen = self.wav2mel(batch.wav_gen)

        if self.use_discriminator:
            self._step_discriminator(batch, is_train)
        self._step_generator(batch, is_train)

        for metric_name in metrics.keys():
            if metric_name.endswith("loss"):
                metrics.update(metric_name, getattr(batch, metric_name).item())

        return batch

    def _step_discriminator(self, batch: Batch, is_train: bool):
        batch.disc_pred, _ = self.discriminator(batch.wav)
        batch.disc_pred_gen, _ = self.discriminator(batch.wav_gen.detach())

        batch.discriminator_loss = self.disc_criterion(batch)

        if is_train:
            self.disc_optimizer.zero_grad()
            batch.discriminator_loss.backward()
            self.disc_optimizer.step()
            if self.disc_scheduler is not None:
                self.disc_scheduler.step()

    def _step_generator(self, batch: Batch, is_train: bool):
        if self.use_discriminator:
            batch.disc_pred, batch.disc_features = self.discriminator(batch.wav)
            batch.disc_pred_gen, batch.disc_features_gen = self.discriminator(batch.wav_gen)

        # correctly calculates loss even if no discriminator
        batch.generator_loss = self.gen_criterion(batch)

        if is_train:
            self.gen_optimizer.zero_grad()
            batch.generator_loss.backward()
            self.gen_optimizer.step()
            if self.gen_scheduler is not None:
                self.gen_scheduler.step()

    @torch.no_grad()
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.generator.eval()
        if self.use_discriminator:
            self.discriminator.eval()
        self.valid_metrics.reset()

        batch = None
        for batch_idx, batch in tqdm(
                enumerate(self.valid_data_loader),
                desc="validation",
                total=len(self.valid_data_loader),
        ):
            batch = self.process_batch(
                batch,
                is_train=False,
                metrics=self.valid_metrics,
            )

        self.writer.set_step(epoch * self.len_epoch, "val")
        self._log_scalars(self.valid_metrics)
        self._log_predictions(batch)

        # add histogram of model parameters to the tensorboard
        for name, p in self.generator.named_parameters():
            self.writer.add_histogram("generator." + name, p, bins="auto")
        if self.use_discriminator:
            for name, p in self.discriminator.named_parameters():
                self.writer.add_histogram("discriminator." + name, p, bins="auto")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def _inference_epoch(self, epoch):
        self.generator.eval()
        self.writer.set_step(epoch * self.len_epoch, "inference")

        for batch_idx, batch in tqdm(
                enumerate(self.inference_loader),
                desc="inference",
                total=len(self.inference_loader),
        ):
            batch = batch.to(self.device)
            batch.wav_gen = self.generator(batch)

            batch.spec_gen = self.wav2mel(batch.wav_gen)

            self._log_predictions(batch, inference_id=batch_idx + 1)

    @torch.no_grad()
    def _log_predictions(
            self,
            batch: Batch,
            inference_id=None
    ):
        if self.writer is None:
            return

        idx = random.randrange(batch.wav_gen.size(0))

        if inference_id is None:
            self._log_audio("true audio", batch.wav[idx])

        name_suffix = str(inference_id) if inference_id is not None else ""

        self._log_spectrogram("true spectrogram" + name_suffix, batch.spec[idx])
        self._log_spectrogram("predicted spectrogram" + name_suffix, batch.spec_gen[idx])
        self._log_audio("generated audio" + name_suffix, batch.wav_gen[idx])

    def _log_spectrogram(self, spec_name, spectrogram):
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram.cpu()))
        self.writer.add_image(spec_name, image)
        image.close()

    def _log_audio(self, audio_name, audio):
        self.writer.add_audio(audio_name, audio.cpu(), self.wav2mel.config.sr)

    @torch.no_grad()
    def get_grad_norm(self, model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
