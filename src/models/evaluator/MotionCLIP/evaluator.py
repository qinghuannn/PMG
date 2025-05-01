import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn

from .motion_clip import MotionCLIP


class MotionCLIPEvaluator(object):
    def __init__(self, dataset='hml3d', deps_dir="./deps/MotionCLIP"):
        self.dataset = dataset
        self.motion_clip = MotionCLIP(motion_dim=263 if self.dataset == 'hml3d' else 251)
        ckpt = torch.load(osp.join(deps_dir, f'{self.dataset}/{self.dataset}.ckpt'), map_location='cpu')
        # self.motion_clip.load_state_dict(ckpt, strict=True)
        self.motion_clip.load_state_dict(ckpt, strict=False)
        self.mean = torch.tensor(np.load(osp.join(deps_dir, f'{self.dataset}/Mean.npy')), dtype=torch.float)
        self.std = torch.tensor(np.load(osp.join(deps_dir, f'{self.dataset}/Std.npy')), dtype=torch.float)

    def extract_embedding(self, motion, motion_len, text):
        text_embed = self.extract_text_embedding(motion.device, text)
        motion_embed = self.extract_motion_embedding(motion, motion_len)
        return motion_embed, text_embed

    @torch.no_grad()
    def extract_text_embedding(self, device, text):
        self.motion_clip.to(device)
        self.motion_clip.eval()
        text_embed = self.motion_clip.encode_text(text, device).float()
        return text_embed / text_embed.norm(dim=1, keepdim=True)

    @torch.no_grad()
    def extract_motion_embedding(self, motion, motion_len):
        self.motion_clip.to(motion.device)
        self.motion_clip.eval()
        device = motion.device
        motion = (motion - self.mean.to(device)) / self.std.to(device)
        motion = motion.float()
        motion_embed = self.motion_clip.encode_motion(motion, motion_len).type(motion.dtype)
        # return motion_embed
        return motion_embed / motion_embed.norm(dim=1, keepdim=True)