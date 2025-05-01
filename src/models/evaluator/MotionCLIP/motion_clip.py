# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn

from transformers import RobertaTokenizer, RobertaModel

class MotionCLIP(nn.Module):
    def __init__(self, motion_dim=263, model_dim=512, output_dim=512,
                 patch_size=1, num_layers=8, num_heads=8, dropout=0.3):
        super(MotionCLIP, self).__init__()

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")
        self.max_text_len = 32
        text_dim = 768

        self.motion_encoder = MotionTransformer(motion_dim, model_dim,
                                                patch_size, num_layers, num_heads, dropout=dropout)

        self.motion_proj = nn.Parameter(output_dim**-0.5 * torch.randn(model_dim, output_dim))
        self.text_proj = nn.Parameter(output_dim**-0.5 * torch.randn(text_dim, output_dim))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_text(self, text, device):
        encoded_input = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
        ).to(device)
        text_feat = self.text_encoder(**encoded_input)['pooler_output']
        text_feat = text_feat @ self.text_proj
        return text_feat

    def encode_motion(self, motion, motion_length):
        x = self.motion_encoder(motion, motion_length)
        x = x @ self.motion_proj
        return x

    def forward(self, motion, text, motion_length):
        motion_feat = self.encode_motion(motion, motion_length)
        text_feat = self.encode_text(text, motion.device)

        motion_feat = motion_feat / motion_feat.norm(dim=1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_motion = logit_scale * motion_feat @ text_feat.t()
        logits_per_text = logits_per_motion.t()

        return logits_per_motion, logits_per_text


class MotionTransformer(nn.Module):
    def __init__(self, motion_dim, model_dim, patch_size, num_layers, num_heads, max_motion_len=256, dropout=0.1):
        super(MotionTransformer, self).__init__()
        scale = model_dim ** -0.5
        self.patch_size = patch_size
        self.conv = nn.Conv1d(in_channels=motion_dim, out_channels=model_dim,
                              kernel_size=patch_size, stride=patch_size, bias=False)
        self.class_token = nn.Parameter(scale * torch.randn(model_dim))
        self.pos_embedding = nn.Parameter(scale * torch.randn(max_motion_len, model_dim))
        self.ln_pre = nn.LayerNorm(model_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                model_dim, num_heads,
                dim_feedforward=model_dim * 4,
                activation='gelu',
                batch_first=True,
                norm_first=True,
                dropout=dropout
            ), num_layers=num_layers
        )
        self.ln_post = nn.LayerNorm(model_dim)

    def forward(self, x, motion_length):
        mask = torch.cat(
            [torch.ones([x.size(0), 1], device=x.device, dtype=torch.bool),
            lengths_to_mask(motion_length // self.patch_size, x.device)],
            dim=1)
        x = self.conv(x.permute([0, 2, 1])).permute([0, 2, 1])
        B, L = x.shape[:2]
        cls_token = self.class_token.repeat([B, 1, 1])
        x = torch.cat([cls_token, x], dim=1)
        # print(mask.shape, x.shape, L)
        x = x + self.pos_embedding[:L+1]
        x = self.ln_pre(x)
        x = self.transformer(x, src_key_padding_mask=~mask)
        x = self.ln_post(x[:, 0])
        return x


def lengths_to_mask(lengths, device):
    """
    Generate mask array.
    """
    lengths = lengths.clone().detach()
    max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask