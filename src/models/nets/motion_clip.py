# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
from transformers import RobertaTokenizer, RobertaModel

from ..utils.utils import lengths_to_mask


class MotionCLIP(nn.Module):
    def __init__(self, motion_dim=263, model_dim=512, output_dim=512,
                 patch_size=1, num_layers=8, num_heads=8, dropout=0.3,
                 pretrain=True, text_encoder='roberta', use_text_former=False):
        super(MotionCLIP, self).__init__()
        self.text_encoder_type = text_encoder
        self.use_text_former = use_text_former
        text_dim = 768

        if text_encoder == 'clip':
            self.text_encoder, _ = clip.load("ViT-B/32", device=torch.device('cpu'), jit=False)
            text_dim = 512
        elif text_encoder == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.text_encoder = RobertaModel.from_pretrained("roberta-base")
            self.max_text_len = 32
            text_dim = 768

        if self.use_text_former:
            self.hidden_proj = nn.Linear(text_dim, model_dim)
            text_dim = model_dim
            self.text_former = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(model_dim, num_heads, dim_feedforward=model_dim * 4, dropout=dropout,
                                           batch_first=True, norm_first=True, activation='gelu'),
                num_layers=num_layers
            )
            self.text_token = nn.Parameter(torch.randn(model_dim))

        if pretrain is False:
            self.text_encoder.initialize_parameters()

        self.motion_encoder = MotionTransformer(motion_dim, model_dim, output_dim,
                                                patch_size, num_layers, num_heads, dropout=dropout)

        self.motion_proj = nn.Parameter(output_dim**-0.5 * torch.randn(model_dim, output_dim))
        self.text_proj = nn.Parameter(output_dim**-0.5 * torch.randn(text_dim, output_dim))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_text(self, text, device):
        if self.text_encoder_type == 'clip':
            text = clip.tokenize(text, truncate=True).to(device)
            if self.use_text_former:
                x = self.text_encoder.token_embedding(text).type(self.text_encoder.dtype)  # [batch_size, n_ctx, d_model]
                x = x + self.text_encoder.positional_embedding.type(self.text_encoder.dtype)
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.text_encoder.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.text_encoder.ln_final(x).type(self.text_encoder.dtype).detach()
                # x.shape = [batch_size, n_ctx, transformer.width]
                # take features from the eot embedding (eot_token is the highest number in each sequence)
                mask = lengths_to_mask(text.argmax(dim=-1), device)
                if mask.shape[1] < x.shape[1]:
                    x = x[:, :mask.size(1)]

                x = self.hidden_proj(x)
                hiddens = torch.cat([self.text_token.repeat([len(text), 1, 1]), x], dim=1)
                mask = torch.cat([torch.ones([len(text), 1], dtype=torch.bool, device=device),
                                  mask.bool()], dim=1)
                text_feat = self.text_former(src=hiddens, src_key_padding_mask=~mask)[:, 0]
            else:
                text_feat = self.text_encoder.encode_text(text)
        elif self.text_encoder_type == 'roberta':
            encoded_input = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
            ).to(device)
            if self.use_text_former:
                with torch.no_grad():
                    output = self.text_encoder(**encoded_input)
                hiddens = self.hidden_proj(output["last_hidden_state"])
                hiddens = torch.cat([self.text_token.repeat([len(text), 1, 1]), hiddens], dim=1)
                mask = torch.cat([torch.ones([len(text), 1], dtype=torch.bool, device=device),
                                  encoded_input["attention_mask"].bool()], dim=1)
                text_feat = self.text_former(src=hiddens, src_key_padding_mask=~mask)[:, 0]
            else:
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

    def freeze(self, motion_encoder=False, text_encoder=False):
        if motion_encoder:
            for param in self.motion_encoder.parameters():
                param.requires_grad = False
        if text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def unfreeze(self, motion_encoder=False, text_encoder=False):
        if motion_encoder:
            for param in self.motion_encoder.parameters():
                param.requires_grad = True
        if text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = True

class MotionTransformer(nn.Module):
    def __init__(self, motion_dim, model_dim, output_dim, patch_size, num_layers, num_heads, max_motion_len=256, dropout=0.1):
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