import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import yaml
from .mobert import MotionTextEvalBERT



class MoBertEvaluator(object):
    def __init__(self, dataset='hml3d', deps_dir="./deps/MoBert"):
        self.dataset = dataset
        # hml3d == t2m
        if dataset != 'hml3d':
            raise ValueError("MoBert only support HumanML3D!")

        with open(f"{deps_dir}/config.yml", "r") as conf:
            config = yaml.safe_load(conf)
        self.model = MotionTextEvalBERT(
            config["primary_evaluator_model"],
            config["chunk_encoder"],
            config["tokenizer_and_embedders"],
            tokenizer_path=f"{deps_dir}/primary_evaluator/std_bpe2000/tokenizer.tk",
            load_trained_regressors_path=f"{deps_dir}/primary_evaluator/std_bpe2000/"
        )
        checkpoint = torch.load(f"{deps_dir}/primary_evaluator/std_bpe2000/best_Faithfulness_checkpoint.pth",
                                map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        self.chunk_size = config["chunk_encoder"]["chunk_size"]
        self.overlap = config["chunk_encoder"]["chunk_overlap"]
        self.pad_len = 1 + (200 // (self.chunk_size + self.overlap))

        self.mean = torch.from_numpy(np.load(f"{deps_dir}/Mean.npy").reshape(1, -1))
        self.std = torch.from_numpy(np.load(f"{deps_dir}/Std.npy").reshape(1, -1))

    @torch.no_grad()
    def calculate_score(self, motion, motion_len, texts):
        self.model.to(motion.device)
        self.model.eval()
        motion_chunks, motion_masks = self.data_process(motion, motion_len)

        alignment, faithfulness_rating, naturalness_rating = self.model.rate_alignment_batch(texts, motion_chunks,
                                                                                             motion_masks,
                                                                                             motion.device)
        return alignment, faithfulness_rating, naturalness_rating

    def data_process(self, data, motion_len):
        self.mean = self.mean.to(data.device)
        self.std = self.std.to(data.device)
        num_chunks = data.shape[1] // self.chunk_size + 1
        # [b, num_chunks+1, d]
        pad_data = torch.cat([data, data[:, :(self.chunk_size - data.shape[1] % self.chunk_size)]], dim=1)
        ndata = pad_data[:, :num_chunks * self.chunk_size].view(data.shape[0], num_chunks, self.chunk_size, -1)
        shift_ndata = ndata.roll(-1, 1)[:, :, :self.overlap]

        motion_chunks = torch.cat([ndata, shift_ndata], dim=2)

        idx = torch.arange(num_chunks, device=data.device).repeat([len(data), 1]) + 1
        motion_masks = (idx * self.chunk_size + self.overlap) < motion_len.unsqueeze(dim=1)
        motion_masks = motion_masks.int()

        if num_chunks < self.pad_len:
            motion_chunks = torch.cat([motion_chunks, torch.zeros(
                [len(data), self.pad_len - num_chunks, motion_chunks.shape[2], motion_chunks.shape[3]],
                device=motion_chunks.device, dtype=motion_chunks.dtype)], dim=1)
            motion_masks = torch.cat([motion_masks, torch.zeros([len(data), self.pad_len - num_chunks],
                                                                device=motion_chunks.device,
                                                                dtype=motion_chunks.dtype)], dim=1)

        motion_chunks = motion_chunks[:, :self.pad_len]
        motion_masks = motion_masks[:, :self.pad_len]

        motion_chunks = (motion_chunks - self.mean) / self.std
        motion_chunks[motion_masks == 0] = 0
        return motion_chunks, motion_masks