import random
from typing import Optional

import PIL
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.logger.utils import plot_spectrogram_to_buf
from src.model import FastSpeech, Waveglow, GraphemeAligner
from src.utils import inf_loop, MetricTracker
from src.utils.config_parser import ConfigParser
from src.utils.data import Batch, MelSpectrogram
from .base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model: FastSpeech,
            wav2mel: MelSpectrogram,
            aligner: GraphemeAligner,
            vocoder: Waveglow,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            config: ConfigParser,
            device: torch.device,
            data_loader: torch.utils.data.DataLoader,
            valid_data_loader=None,
            inference_data_loader=None,
            lr_scheduler=None,
            len_epoch: Optional[int] = None,
            skip_oom: bool = True,
            log_step: Optional[int] = None
    ):
        super().__init__(model, criterion, optimizer, config, device)
        self.wav2mel = wav2mel
        self.aligner = aligner
        self.vocoder = vocoder

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

        self.lr_scheduler = lr_scheduler
        self.log_step = log_step if log_step is not None else 100

        self.train_metrics = MetricTracker(
            "loss", "mel_loss", "duration_loss", "grad norm", writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", "mel_loss", "duration_loss", writer=self.writer
        )

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
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
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch.loss.item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
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
            batch.mels = self.wav2mel(batch.waveform)
            batch.mels_length = self.wav2mel.transform_wav_lengths(batch.waveform_length)

            aligner_durations = (self.aligner(batch.waveform, batch.waveform_length, batch.transcript)
                                 .to(self.device)
                                 * batch.mels_length.unsqueeze(1))

        if is_train:
            batch.durations = aligner_durations
            self.optimizer.zero_grad()

        mels, log_lengths, mels_lens = self.model(batch)

        if not is_train:
            # simulate inference
            batch.durations = aligner_durations

        batch.mels_pred = mels
        batch.mels_pred_length = mels_lens
        batch.durations_pred = log_lengths.exp()

        batch.mel_loss, batch.dur_loss = self.criterion(batch)
        batch.loss = batch.mel_loss + batch.dur_loss

        if is_train:
            batch.loss.backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        metrics.update("loss", batch.loss.item())
        metrics.update("mel_loss", batch.mel_loss.item())
        metrics.update("duration_loss", batch.dur_loss.item())

        return batch

    @torch.no_grad()
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
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
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
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
        self.model.eval()
        self.writer.set_step(epoch * self.len_epoch, "inference")

        for batch_idx, batch in tqdm(
                enumerate(self.inference_loader),
                desc="inference",
                total=len(self.inference_loader),
        ):
            batch = batch.to(self.device)
            mels, _, mels_lens = self.model(batch)

            batch.mels_pred = mels
            batch.mels_pred_length = mels_lens

            self._log_predictions(batch, inference_id=batch_idx + 1)

    @torch.no_grad()
    def _log_predictions(
            self,
            batch: Batch,
            inference_id=None
    ):
        if self.writer is None:
            return

        idx = random.randrange(len(batch.transcript))

        if inference_id is None:
            self._log_spectrogram("true spectrogram", batch.mels[idx])
            self._log_audio("true audio", batch.waveform[idx, :batch.waveform_length[idx]])

        name_suffix = str(inference_id) if inference_id is not None else ""
        self.writer.add_text("transcript" + name_suffix, batch.transcript[idx])
        self._log_spectrogram("predicted spectrogram" + name_suffix, batch.mels_pred[idx])
        self._log_audio("generated audio" + name_suffix, self.vocoder.inference(
            batch.mels_pred[idx, :, :batch.mels_pred_length[idx]].unsqueeze(0)
        ).squeeze())

    def _log_spectrogram(self, spec_name, spectrogram):
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram.cpu()))
        self.writer.add_image(spec_name, image)
        image.close()

    def _log_audio(self, audio_name, audio):
        self.writer.add_audio(audio_name, audio.cpu(), self.wav2mel.config.sr)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
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
