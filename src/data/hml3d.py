
from typing import Optional, Callable
import os.path as osp

from omegaconf import DictConfig
from hydra.utils import get_original_cwd

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
# from lightning_utilities.core.rank_zero import rank_zero_only
from torch.utils.data import DataLoader, Dataset

from .utils import mld_collate
from .humanml.dataset import Text2MotionDatasetV2, Text2MotionDatasetV3


class HumanML3DDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        val_batch_size: int = -1,
        test_batch_size: int = 1,
        num_workers: int = 16,
        pin_memory: bool = False,
        max_motion_length: int = 196,
        min_motion_length: int = 40,
        max_text_len: int = 20,
        unit_length: int = 4,
        keyframe_info=None,
        norm_method: bool = True,
        w_vectorizer_path: str = '',
        dataset_name: str = 'hml3d',
        no_aug: bool = False,
        repeat_dataset=1,
        scale=1,
        motion_repr="hml3d",

        njoints: int = 22,
        motion_dim: int = 263,

    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        if dataset_name == "hml3d":
            self.data_dir = osp.join(data_dir, "HumanML3D")
        else:
            self.data_dir = osp.join(data_dir, "KIT-ML")
        self.njoints = njoints
        self.dataloader_options = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": False,
            "collate_fn": mld_collate
        }

        self.name = dataset_name
        self.dataset = Text2MotionDatasetV2
        self.norm_method = norm_method
        self.w_vectorizer_path = w_vectorizer_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: None):
        self.hparams.keyframe_info.file = osp.join(self.data_dir, self.hparams.keyframe_info.file)
        self.dataset_kwargs = {
            "dataset_name": self.hparams.dataset_name,
            "data_dir": self.data_dir,
            "w_vectorizer_path": self.w_vectorizer_path,
            "max_motion_length": self.hparams.max_motion_length,
            "min_motion_length": self.hparams.min_motion_length,
            "max_text_len": self.hparams.max_text_len,
            "unit_length": self.hparams.unit_length,
            "norm_method": self.norm_method,
            "keyframe_info": self.hparams.keyframe_info,
            "motion_repr": self.hparams.motion_repr,
            "scale": self.hparams.scale,
        }

    def train_dataloader(self):
        if self.train_dataset is None:
            self.train_dataset = self.dataset(split_file="train.txt",
                                              repeat_dataset=self.hparams.repeat_dataset,
                                              **self.dataset_kwargs)
            self.nfeats = self.train_dataset.nfeats
        options = self.dataloader_options.copy()
        options["batch_size"] = self.hparams.batch_size
        return DataLoader(dataset=self.train_dataset, shuffle=True, **options)

    def val_dataloader(self):
        if self.val_dataset is None:
            self.val_dataset = self.dataset(split_file="val.txt", no_aug=self.hparams.no_aug,
                                                    **self.dataset_kwargs)
        options = self.dataloader_options.copy()
        options["batch_size"] = self.hparams.val_batch_size
        if options["batch_size"] == -1:
            options["batch_size"] = self.hparams.batch_size
        return DataLoader(dataset=self.val_dataset, shuffle=False, drop_last=False, **options)

    def test_dataloader(self):
        if self.test_dataset is None:
            self.test_dataset = self.dataset(split_file="test.txt", no_aug=self.hparams.no_aug,
                                                     **self.dataset_kwargs)
            self.nfeats = self.test_dataset.nfeats
            self.test_dataset.is_mm = False

        options = self.dataloader_options.copy()
        options["batch_size"] = self.hparams.test_batch_size

        return DataLoader(dataset=self.test_dataset, shuffle=True, drop_last=False, **options)
        # return DataLoader(dataset=self.test_dataset, shuffle=False, drop_last=True, **options)

    def mm_mode(self, mm_on=True, mm_num_samples=100):
        # random select samples for mm
        if mm_on:
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list, mm_num_samples, replace=False)
            self.test_dataset.name_list = self.mm_list
            self.test_dataset.is_mm = True
        else:
            self.test_dataset.is_mm = False
            self.test_dataset.name_list = self.name_list


class HumanML3DKeyframeDataModule(HumanML3DDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        val_batch_size: int = -1,
        test_batch_size: int = 1,
        num_workers: int = 16,
        pin_memory: bool = False,
        max_motion_length: int = 196,
        min_motion_length: int = 40,
        max_text_len: int = 20,
        unit_length: int = 4,
        njoints: int = 22,
        motion_dim: int = 263,
        keyframe_info=None,
        norm_method: bool = True,
        w_vectorizer_path: str = '',
        dataset_name: str = 'hml3d',
        flip_keyframe=True,
        repeat_dataset=1,
        scale=1,
        motion_repr="hml3d",
    ):
        super().__init__(data_dir=data_dir, batch_size=batch_size, val_batch_size=val_batch_size,
                         test_batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory,
                         max_motion_length=max_motion_length, min_motion_length=min_motion_length,
                         max_text_len=max_text_len, unit_length=unit_length, njoints=njoints,
                         motion_dim=motion_dim, keyframe_info=keyframe_info, norm_method=norm_method,
                         w_vectorizer_path=w_vectorizer_path, dataset_name=dataset_name,repeat_dataset=repeat_dataset,
                         scale=scale, motion_repr=motion_repr)
        self.flip_keyframe = flip_keyframe
        self.dataset = Text2MotionDatasetV3

    def setup(self, stage: None):
        self.hparams.keyframe_info.file = osp.join(self.data_dir, self.hparams.keyframe_info.file)
        self.dataset_kwargs = {
            "dataset_name": self.hparams.dataset_name,
            "data_dir": self.data_dir,
            "w_vectorizer_path": self.w_vectorizer_path,
            "max_motion_length": self.hparams.max_motion_length,
            "min_motion_length": self.hparams.min_motion_length,
            "max_text_len": self.hparams.max_text_len,
            "unit_length": self.hparams.unit_length,
            "norm_method": self.norm_method,
            "keyframe_info": self.hparams.keyframe_info,
            "motion_repr": self.hparams.motion_repr,
            "scale": self.hparams.scale,
            "flip_keyframe": self.flip_keyframe
        }





if __name__ == '__main__':
    datamodule = HumanML3DDataModule("./data")
    datamodule.setup(None)
    dataloader = datamodule.test_dataloader()
    for i, data in enumerate(dataloader):
        # motion, text, length, word_embs, pos_ohot, text_len, tokens = data
        print(data["motion"].shape, data["text"], data["length"], data["text_len"])
        break