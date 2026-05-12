# Copyright (c) OpenMMLab. All rights reserved.
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
        """
        Scale Embeddings module
        """
        assert len(local_crop_size) == 2, "local_crop_size should be a tuple of (H, W)"
        super(ScaleEmbeddings, self).__init__()
        base_embedding = nn.Parameter(
            torch.randn(local_crop_size)
        )
        
        nn.init.trunc_normal_(base_embedding, std=0.02)

        self.scale_embed = base_embedding.unsqueeze(0).unsqueeze(0) # 1,1,H,W
        self.patch_size = local_crop_size
        
    def forward(self, features):
        B,C = features.shape[0], features.shape[1]
        embed = self.scale_embed.repeat(B,C,1,1) # B,C,H,W

        assert features.shape == embed.shape, \
            f"feature size {features.shape} and embeddings {embed.shape} do not match"

        return features + embed.to(features.device)

@MODELS.register_module()
class SFRNet(BaseModule):
    """
    Scale-Frustum Representation Network
    """

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
        
        elif depth[0] in [1]: # swin tiny
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

        elif depth[0] in [2]: # swin small
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

        elif depth[0] in [3]: # swin base
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
        
        elif depth[0] in [4]: # swin large
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

    def forward_trace(self, x):
        """Forward function."""
        outs = []
        split_x = torch.split(x, 3, dim=1)
        for i in range(len(split_x)):
            if i == 0:
                x = self.scale_embedding0(split_x[i])

                # for ab_1
                # x = split_x[i]

                for main_feat in self.main_encoder(x):
                    outs.append(main_feat)
                    
            elif i == 1:
                x = self.scale_embedding1(split_x[i])

                # for ab_1
                # x = split_x[i]

                outs.append(self.sub_encoder1(x)[0])

            elif i == 2:
                x = self.scale_embedding2(split_x[i])

                # for ab_1
                # x = split_x[i]

                outs.append(self.sub_encoder2(x)[0])
        
        return tuple(outs)
    
    def forward(self, x):
        """
        Trace-friendly forward function.
        x is assumed to be a (B, 9, H, W) tensor where 9 = 3_local + 3_short + 3_long.
        """
        # 1. 使用静态的张量切片 (slicing) 替代 torch.split
        # 假设 9 通道总是按 Local, Short, Long 的顺序堆叠
        x_local = x[:, 0:3, :, :]
        x_short = x[:, 3:6, :, :]
        x_long  = x[:, 6:9, :, :]

        outs = []

        # 2. 移除循环，显式调用每个编码器
        
        # Process Local (Main Encoder)
        x_local_emb = self.scale_embedding0(x_local)
        # main_encoder 输出 4 个 stage 的 features
        main_feats = self.main_encoder(x_local_emb) 
        outs.append(main_feats[0])
        outs.append(main_feats[1])
        outs.append(main_feats[2])
        outs.append(main_feats[3])

        # Process Short (Sub Encoder 1)
        x_short_emb = self.scale_embedding1(x_short)
        # sub_encoder1 只输出 stage 3 的 feature (在 __init__ 中定义)
        sub1_feat = self.sub_encoder1(x_short_emb) 
        outs.append(sub1_feat[0]) # 假设 sub_encoder1(x)[0] 是您想要的

        # Process Long (Sub Encoder 2)
        x_long_emb = self.scale_embedding2(x_long)
        # sub_encoder2 只输出 stage 3 的 feature (在 __init__ 中定义)
        sub2_feat = self.sub_encoder2(x_long_emb) 
        outs.append(sub2_feat[0]) # 假设 sub_encoder2(x)[0] 是您想要的

        return tuple(outs)
    
