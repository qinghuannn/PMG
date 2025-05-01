# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import MeanMetric, MetricCollection
# import pytorch_lightning as pl
import lightning.pytorch as pl
from hydra.utils import get_original_cwd

from .metrics import CLIPMetric


from .utils.utils import lengths_to_mask, occumpy_mem, CosineWarmupScheduler, SimpleDict


class MotionCLIP(pl.LightningModule):
    def __init__(self,
                 motion_clip,
                 optimizer,
                 lr_scheduler=False,
                 warmup=0,
                 only_train_mtenc=0,
                 ocpm=False,
                 compile=False,
                 **kwargs
        ):
        super(MotionCLIP, self).__init__()
        self.save_hyperparameters(logger=False, ignore=['motion_clip'])
        self.motion_clip = motion_clip
        self.configure_metrics()

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            self.motion_clip = torch.compile(self.motion_clip)

    def configure_optimizers(self):
        # self.motion_clip.text_encoder.require_grad = False
        # optimizer = self.hparams.optimizer(filter(lambda p: p.requires_grad, self.motion_clip.parameters()))
        optimizer = self.hparams.optimizer(self.motion_clip.parameters())
        # optimizer = self.hparams.optimizer([
        #             {'params': self.motion_clip.motion_encoder.parameters()},
        #             {'params': self.motion_clip.motion_proj},
        #             {'params': self.motion_clip.logit_scale},
        #             # {'params': self.motion_clip.text_encoder.parameters(), 'lr': 0.0001*0.01}])
        #             {'params': self.motion_clip.text_encoder.parameters(), 'lr': 0.0001*0.01},
        #             {'params': self.motion_clip.text_proj},
        # ])
        if self.hparams.lr_scheduler:
            # lr_scheduler = self.hparams.lr_scheduler(optimizer=optimizer)
            lr_scheduler = CosineWarmupScheduler(optimizer, T_max=self.trainer.estimated_stepping_batches,
                                                 eta_min=1e-6, warmup=self.hparams.warmup)
            lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        return optimizer

    def configure_metrics(self):
        metrics = {f"{split}_loss": MeanMetric() for split in ['train', 'val', 'test']}
        metrics.update({
            "eval_r_precision": CLIPMetric(diversity_times=250),
            "test_r_precision": CLIPMetric(),
        })
        self.metrics = MetricCollection(metrics)

    def on_train_start(self):
        if self.hparams.ocpm:
            occumpy_mem(self.device.index)
        for key in self.metrics:
            self.metrics[key].reset()

    def _step_network(self, split, batch, batch_idx):
        motion, text, length = batch["motion"], batch["text"], batch["length"]
        logits_per_motion, logits_per_text = self.motion_clip(motion, text, length)
        labels = torch.arange(motion.shape[0], device=self.device).long()
        loss = F.cross_entropy(logits_per_motion, labels) + F.cross_entropy(logits_per_text, labels)
        loss *= 0.5
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step_network('train', batch, batch_idx)
        self.metrics['train_loss'](loss)
        self.log('Loss/train', self.metrics['train_loss'], prog_bar=True)
        return loss

    def on_train_epoch_start(self) -> None:
        if self.hparams.only_train_mtenc > self.current_epoch:
            self.motion_clip.freeze(text_encoder=True)
        elif self.hparams.only_train_mtenc != 0:
            self.motion_clip.unfreeze(text_encoder=True)

    def on_train_epoch_end(self):
        self.metrics['train_loss'].reset()

    def validation_step(self, batch, batch_idx):
        loss = self._step_network('val', batch, batch_idx)
        self.metrics['val_loss'](loss)
        self.log('Loss/val', self.metrics['val_loss'], sync_dist=True, prog_bar=True)
        text_feat = self.motion_clip.encode_text(batch['text'], self.device)
        motion_feat = self.motion_clip.encode_motion(batch['motion'], batch['length'])
        self.metrics['eval_r_precision'].update(text_feat, motion_feat)

    def on_validation_epoch_end(self):
        results = {}
        metric_output = self.metrics['eval_r_precision'].compute(sanity_flag=self.trainer.sanity_checking)
        self.metrics['eval_r_precision'].reset()
        results.update({
            f"Metrics/eval_{metric}": value.item()
            for metric, value in metric_output.items()
        })
        results.update({
            "epoch": self.trainer.current_epoch,
        })
        self.log_dict(results, sync_dist=True, prog_bar=False)

    def on_test_epoch_begin(self):
        self.motion_clip.eval()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        text_feat = self.motion_clip.encode_text(batch['text'], self.device)
        motion_feat = self.motion_clip.encode_motion(batch['motion'], batch['length'])
        self.metrics['test_r_precision'].update(text_feat, motion_feat)

    def on_test_epoch_end(self):
        results = {}
        metric_output = self.metrics['test_r_precision'].compute(sanity_flag=self.trainer.sanity_checking)
        self.metrics['test_r_precision'].reset()
        results.update({
            f"Metrics/test_{metric}": value.item()
            for metric, value in metric_output.items()
        })
        self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

