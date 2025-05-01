# -*- coding: utf-8 -*-
"""
 @File    : clip.py
 @Time    : 2023/4/26 11:01
 @Author  : Ling-An Zeng
 @Email   : linganzeng@gmail.com
 @Software: PyCharm
"""
from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance

from .utils import *


class CLIPMetric(Metric):
    full_state_update = True

    def __init__(self,
                 top_k=3,
                 R_size=32,
                 dist_sync_on_step=True,
                 diversity_times=300,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.top_k = top_k
        self.R_size = R_size
        self.diversity_times = diversity_times

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        self.metrics = []
        for k in range(1, top_k + 1):
            self.add_state(
                f"R_precision_top_{str(k)}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.metrics.append(f"R_precision_top_{str(k)}")

        # Diversity
        self.add_state("Diversity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("Diversity")

        # chached batches
        self.add_state("text_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("motion_embeddings", default=[], dist_reduce_fx=None)

    def compute(self, sanity_flag):
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # if in sanity check stage then jump
        if sanity_flag:
            return metrics

        texts = torch.cat(self.text_embeddings, dim=0)
        motions = torch.cat(self.motion_embeddings, dim=0)
        # cat all embeddings
        count_seq = texts.shape[0]
        shuffle_idx = torch.randperm(count_seq)


        texts = texts[shuffle_idx]
        motions = motions[shuffle_idx]

        device = motions.device
        # Compute r-precision
        assert count_seq > self.R_size
        top_k_mat = torch.zeros((self.top_k, ), device=device)
        for i in range(count_seq // self.R_size):
            group_texts = texts[i * self.R_size:(i + 1) * self.R_size]
            group_motions = motions[i * self.R_size:(i + 1) * self.R_size]
            dist_mat = euclidean_distance_matrix(group_texts, group_motions).nan_to_num()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
        R_count = count_seq // self.R_size * self.R_size
        for k in range(self.top_k):
            metrics[f"R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        motions = motions.cpu().numpy()

        # Compute diversity
        assert count_seq > self.diversity_times
        metrics["Diversity"] = calculate_diversity_np(motions, self.diversity_times)

        return {**metrics}

    def update(
        self,
        text_embeddings: Tensor,
        motion_embeddings: Tensor,
    ):
        self.text_embeddings.append(text_embeddings)
        self.motion_embeddings.append(motion_embeddings)