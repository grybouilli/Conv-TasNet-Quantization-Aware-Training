import pytorch_lightning as pl
import torch
import torch.nn as nn


class ConvTasNetLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        super().__init__()
        # We should ignore the hyperparameters of the model and loss, as they are complete Pytorch objects
        # self.save_hyperparameters()

        self.model = model
        self.loss = loss

        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def common_step(self, batch, batch_idx):
        inputs, targets = batch
        est_targets = self(inputs)
        loss = self.loss(est_targets, targets)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("loss", loss, logger=True)

        # self.log(
        #     "encoder_scale",
        #     self.model.module.encoder.conv1d.activation_quantizer.scale,
        #     logger=True,
        # )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("memory_allocated (MB)", torch.cuda.memory_allocated() / 1024**2)
        # print(f"Validation step {batch_idx} - loss: {loss.item():.4f}, memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # print(print(torch.cuda.memory_summary(device=None, abbreviated=False)))
        # for name, param in self.model.get_loggable_weights():
        #     self.logger.experiment.add_histogram(name, param, global_step=self.current_epoch)

    def configure_model(self):
        self.model = torch.compile(self.model)
        return

    def configure_optimizers(self):
        if self.scheduler is not None:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.scheduler,
                "monitor": "val_loss",
            }

        return self.optimizer
