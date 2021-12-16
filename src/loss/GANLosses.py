import torch
from torch import nn

from src.utils.data import Batch


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, batch: Batch):
        assert batch.disc_pred is not None
        assert batch.disc_pred_gen is not None

        loss_real = sum(self.mse(d_out, torch.ones_like(d_out)) for d_out in batch.disc_pred)

        # d_out here should already be detached
        loss_gen = sum(self.mse(d_out, torch.zeros_like(d_out)) for d_out in batch.disc_pred_gen)

        batch.discriminator_loss = loss_real + loss_gen

        return batch.discriminator_loss


class GeneratorLoss(nn.Module):
    def __init__(self, feature_coef=2.0, mel_coef=45.0, use_discriminator=True):
        super().__init__()

        self.feature_coef = feature_coef
        self.mel_coef = mel_coef
        self.use_discriminator = use_discriminator

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, batch: Batch):
        assert batch.spec is not None
        assert batch.spec_gen is not None
        batch.spec_loss = self.l1(batch.spec_gen, batch.spec)

        if not self.use_discriminator:
            # return normalized
            return batch.spec_loss

        batch.spec_loss = batch.spec_loss * self.mel_coef

        assert batch.disc_pred_gen is not None
        batch.adv_gen_loss = sum(self.mse(d_out, torch.ones_like(d_out)) for d_out in batch.disc_pred_gen)

        assert batch.disc_features is not None
        assert batch.disc_features_gen is not None
        batch.feature_loss = sum(self.l1(feature_real, feature_gen) for (feature_real, feature_gen) in
                                 zip(batch.disc_features, batch.disc_features_gen)) * self.feature_coef

        batch.generator_loss = (batch.adv_gen_loss
                                + batch.feature_loss
                                + batch.spec_loss)

        return batch.generator_loss
