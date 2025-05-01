# -*- coding: utf-8 -*-
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

from src.models.utils.embedding import timestep_embedding, TimestepEmbedding, PositionEmbedding, PartialPositionEmbedding
from src.models.utils.utils import lengths_to_mask
from .modules import CDGDecoderLayer, TransformerDecoder


class Denoiser(nn.Module):
    def __init__(
        self,
        motion_dim=263,
        model_dim=512,
        text_feat_dim=512,
        time_embed_dim=256,
        max_motion_length=256,
        max_text_length=128,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_layers: int = 4,
        pos_emb: str = 'cos',
        gm_proj: bool = True,
        norm_first: bool = False,
        input_norm: bool = True,
        c: int = 4,
        use_mask_token: bool = True,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.pos_emb = pos_emb
        self.c = c
        if type(num_layers) is str:
            m_layer, t_layer = [int(x) for x in num_layers.split('-')]
        else:
            m_layer, t_layer = num_layers, num_layers
        if pos_emb == 'cos':
            self.m_pos_emb = PartialPositionEmbedding(max_motion_length + 4, model_dim, 0, grad=False)
            self.t_pos_emb = PartialPositionEmbedding(max_text_length + 4, model_dim, 0, grad=False)
        elif pos_emb == 'learn':
            self.m_pos_emb = PositionEmbedding(max_motion_length, model_dim, 0, grad=True, randn_norm=True)
            self.t_pos_emb = PositionEmbedding(max_text_length, model_dim, 0, grad=True, randn_norm=True)
        else:
            raise ValueError(f"Unsupported : {pos_emb}")

        self.time_emb = TimestepEmbedding(time_embed_dim, model_dim)

        self.text_proj = nn.Linear(text_feat_dim, model_dim)
        self.text_former = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(model_dim, num_heads, dim_feedforward,
                                           dropout, activation=F.gelu, batch_first=True, norm_first=norm_first),
                num_layers=t_layer
            )


        self.motion_former = TransformerDecoder(
            CDGDecoderLayer(model_dim, num_heads, dim_feedforward,
                            dropout, activation=F.gelu, batch_first=True, norm_first=norm_first,
                            c=self.c),
            num_layers=m_layer
        )

        self.input_proj = nn.Linear(motion_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, motion_dim)

        if gm_proj:
            self.gm_proj = nn.Linear(motion_dim, model_dim)
        else:
            self.gm_proj = None

        if input_norm:
            self.pm_norm = nn.LayerNorm(model_dim)
            self.gm_norm = nn.LayerNorm(model_dim)
        else:
            self.pm_norm = nn.Identity()
            self.gm_norm = nn.Identity()
        self.use_mask_token = use_mask_token
        if self.use_mask_token:
            self.no_prior_token = nn.Parameter(torch.randn(motion_dim), requires_grad=True)
        self.init_prior_proj = nn.Linear(motion_dim, model_dim)
        # self.padding_token = nn.Parameter(torch.randn(model_dim), requires_grad=False)

    def forward(self, text, timestep, pmotion, ppos, pmask, gmotion, gpos, gmask, motion_prior):
        device = pmotion.device
        B = len(pmotion)

        pmotion[pmask] = 0
        gmotion[gmask] = 0

        # Add a specific token for data without given frames
        if self.use_mask_token:
            no_prior_sample = (~gmask).sum(dim=-1) == 0
            gmotion[no_prior_sample, 0] = self.no_prior_token
            gmask[no_prior_sample, 0] = False

        pmotion = self.input_proj(pmotion)
        _gmotion = self.init_prior_proj(gmotion)
        init_prior_mask = (motion_prior.gather(dim=1, index=gpos) == 1).unsqueeze(dim=-1)

        if self.gm_proj is not None:
            gmotion = self.gm_proj(gmotion)
        else:
            gmotion = self.input_proj(gmotion)
        gmotion = _gmotion * init_prior_mask + gmotion * (~init_prior_mask)

        time_embed = self.time_emb(timestep_embedding(timestep, self.time_embed_dim)).unsqueeze(dim=1)

        new_pmotion = self.m_pos_emb(torch.cat([time_embed, pmotion], dim=1),
                                     torch.cat([torch.zeros([B, 1], dtype=int, device=device), ppos+1], dim=1))
        new_gmotion = self.m_pos_emb(torch.cat([time_embed, gmotion], dim=1),
                                     torch.cat([torch.zeros([B, 1], dtype=int, device=device), gpos+1], dim=1))
        new_pmask = torch.cat([torch.zeros([B, 1], dtype=bool, device=device), pmask], dim=1)
        new_gmask = torch.cat([torch.zeros([B, 1], dtype=bool, device=device), gmask], dim=1)

        new_pmotion = self.pm_norm(new_pmotion)
        new_gmotion = self.gm_norm(new_gmotion)

        text_feat = self.text_proj(text['hidden'])
        text_mask = text['mask']
        text_feat = self.text_former(tgt=text_feat, tgt_key_padding_mask=~text_mask,
                                    memory=new_gmotion, memory_key_padding_mask=new_gmask)

        out = self.motion_former(tgt=new_pmotion, tgt_key_padding_mask=new_pmask,
                                    memory=text_feat, memory_key_padding_mask=~text_mask,
                                    gm=new_gmotion, gm_key_padding_mask=new_gmask)

        out = out[:, 1:]
        out = self.output_proj(out)
        return out