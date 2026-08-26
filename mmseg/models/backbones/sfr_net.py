
import warnings
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmseg.registry import MODELS
from .resnet import ResNetV1c
from .swin import SwinTransformer

class ScaleEmbeddings(nn.Module):
    def __init__(self, local_crop_size=(512,512)):

        assert len(local_crop_size) == 2, "local_crop_size should be a tuple of (H, W)"
        super(ScaleEmbeddings, self).__init__()
        self.scale_embed = nn.Parameter(
            torch.empty(1, 1, *local_crop_size)
        )
        nn.init.trunc_normal_(self.scale_embed, std=0.02)
        self.patch_size = local_crop_size

    def forward(self, features):
        B,C = features.shape[0], features.shape[1]
        embed = self.scale_embed.expand(B,C,-1,-1)

        assert features.shape == embed.shape, \
            f"feature size {features.shape} and embeddings {embed.shape} do not match"

        return features + embed

@MODELS.register_module()
class SFRNet(BaseModule):

    def __init__(self,
                 depth = [50,18,18],
                 local_crop_size = (512,512),
                 init_cfg=None):
        super().__init__(init_cfg)

        depth2ckpt = {
            18: 'pretrain/resnet18_v1c-b5776b93.pth',
            4: 'pretrain/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth',
        }

        if depth[0] in [18, 50, 101]:
            self.main_encoder = ResNetV1c(
                depth=depth[0],
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 2, 4),
                strides=(1, 2, 1, 1),
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True,
                init_cfg=dict(type='Pretrained', checkpoint=depth2ckpt[depth[0]]))

        elif depth[0] in [1]:
            self.main_encoder = SwinTransformer(
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                patch_norm=True,
                out_indices=(0,1,2,3,),
                with_cp=False,
                frozen_stages=-1,
                init_cfg=dict(type='Pretrained', checkpoint=depth2ckpt[depth[0]]))

        elif depth[0] in [2]:
            self.main_encoder = SwinTransformer(
                embed_dims=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                patch_norm=True,
                out_indices=(0,1,2,3,),
                with_cp=False,
                frozen_stages=-1,
                init_cfg=dict(type='Pretrained', checkpoint=depth2ckpt[depth[0]]))

        elif depth[0] in [3]:
            self.main_encoder = SwinTransformer(
                pretrain_img_size=384,
                embed_dims=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=12,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                patch_norm=True,
                out_indices=(0,1,2,3,),
                with_cp=False,
                frozen_stages=-1,
                init_cfg=dict(type='Pretrained', checkpoint=depth2ckpt[depth[0]]))

        elif depth[0] in [4]:
            self.main_encoder = SwinTransformer(
                pretrain_img_size=384,
                embed_dims=192,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                window_size=12,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                with_cp=False,
                frozen_stages=-1,
                init_cfg=dict(type='Pretrained', checkpoint=depth2ckpt[depth[0]]))

        if depth[1] in [18, 50, 101]:
            self.sub_encoder1 = ResNetV1c(
                depth=depth[1],
                num_stages=4,
                out_indices=(3, ),
                dilations=(1, 1, 2, 4),
                strides=(1, 2, 1, 1),
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True,
                init_cfg=dict(type='Pretrained', checkpoint=depth2ckpt[depth[1]]))

            self.sub_encoder2 = ResNetV1c(
                depth=depth[2],
                num_stages=4,
                out_indices=(3,),
                dilations=(1, 1, 2, 4),
                strides=(1, 2, 1, 1),
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True,
                init_cfg=dict(type='Pretrained', checkpoint=depth2ckpt[depth[2]]))

        elif depth[1] in [1]:
            self.sub_encoder1 = SwinTransformer(
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                patch_norm=True,
                out_indices=(3,),
                with_cp=False,
                frozen_stages=-1,
                init_cfg=dict(type='Pretrained', checkpoint=depth2ckpt[depth[1]]))

            self.sub_encoder2 = SwinTransformer(
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                patch_norm=True,
                out_indices=(3,),
                with_cp=False,
                frozen_stages=-1,
                init_cfg=dict(type='Pretrained', checkpoint=depth2ckpt[depth[1]]))

        elif depth[1] in [2]:
            self.sub_encoder1 = SwinTransformer(
                embed_dims=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                patch_norm=True,
                out_indices=(3,),
                with_cp=False,
                frozen_stages=-1,
                init_cfg=dict(type='Pretrained', checkpoint=depth2ckpt[depth[1]]))

            self.sub_encoder2 = SwinTransformer(
                embed_dims=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                patch_norm=True,
                out_indices=(3,),
                with_cp=False,
                frozen_stages=-1,
                init_cfg=dict(type='Pretrained', checkpoint=depth2ckpt[depth[1]]))

        self.scale_embedding0 = ScaleEmbeddings(local_crop_size=local_crop_size)
        self.scale_embedding1 = ScaleEmbeddings(local_crop_size=local_crop_size)
        self.scale_embedding2 = ScaleEmbeddings(local_crop_size=local_crop_size)

    def forward(self, x):

        x_local = x[:, 0:3, :, :]
        x_short = x[:, 3:6, :, :]
        x_long  = x[:, 6:9, :, :]
        outs = []

        x_local_emb = self.scale_embedding0(x_local)

        main_feats = self.main_encoder(x_local_emb)
        outs.append(main_feats[0])
        outs.append(main_feats[1])
        outs.append(main_feats[2])
        outs.append(main_feats[3])

        x_short_emb = self.scale_embedding1(x_short)

        sub1_feat = self.sub_encoder1(x_short_emb)
        outs.append(sub1_feat[0])


        x_long_emb = self.scale_embedding2(x_long)

        sub2_feat = self.sub_encoder2(x_long_emb)
        outs.append(sub2_feat[0])

        return tuple(outs)

