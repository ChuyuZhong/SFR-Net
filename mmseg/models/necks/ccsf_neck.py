# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch
import numpy as np
from mmengine.model.weight_init import constant_init
from torch.nn import functional as F
from mmcv.cnn import ConvModule, build_norm_layer
import cv2
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from mmseg.registry import MODELS

class LowRankAttention(nn.Module):
    """
    Low-Rank Attention Module
    copied and modified from ..utils.SelfAttentionBlock
    """

    def __init__(self, channels,
                 share_key_query=False,
                 query_downsample=None,
                 key_downsample=None,
                 key_query_num_convs=2,
                 value_out_num_convs=2,
                 key_query_norm=True,
                 value_out_norm=True,
                 matmul_norm=True,
                 with_out=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 act_cfg=None):
        key_in_channels = channels // 2
        query_in_channels = channels
        mid_channels = channels // 4
        out_channels = channels
        
        super().__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = mid_channels
        self.share_key_query = share_key_query
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        
        self.key_project = self.build_project(
            key_in_channels,
            mid_channels,
            num_convs=key_query_num_convs,
            use_conv_module=key_query_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(
                query_in_channels,
                mid_channels,
                num_convs=key_query_num_convs,
                use_conv_module=key_query_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        
        self.value_project = self.build_project(
            key_in_channels,
            mid_channels if with_out else out_channels,
            num_convs=value_out_num_convs,
            use_conv_module=value_out_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        if with_out:
            self.out_project = self.build_project(
                mid_channels,
                out_channels,
                num_convs=value_out_num_convs,
                use_conv_module=value_out_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.out_project = None

        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm

        self.init_weights()

    def init_weights(self):
        """Initialize weight of later layer."""
        if self.out_project is not None:
            if not isinstance(self.out_project, ConvModule):
                constant_init(self.out_project, 0)

    def build_project(self, in_channels, channels, num_convs, use_conv_module,
                      conv_cfg, norm_cfg, act_cfg):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats):
        """Forward function with skip connection."""
        identity = query_feats
        
        batch_size = query_feats.size(0)
        
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()


        # print(query.shape, key.shape, value.shape)

        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        
        if self.out_project is not None:
            context = self.out_project(context)
        
        context = context + identity
        
        return context, sim_map

class MLPAlign(nn.Module):
    """
    simple MLP to align the channels of different features
    """
    def __init__(self, input_channels, output_channels, mlp_ratio , act_cfg=nn.ReLU()):
        super().__init__()
        hidden_channels = int(input_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            act_cfg,
            nn.Conv2d(hidden_channels, output_channels, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        return self.mlp(x)
    

def save_attention_heatmap(sim_map, index, suffix):
    """优化后的注意力热力图保存函数，增强视觉对比度"""
    batch_size = sim_map.shape[0]
    query_H, query_W = 16, 16
    key_H, key_W = 64, 64

    # 重塑注意力图为空间维度 (B, query_H, query_W, key_H, key_W)
    # sim_map_reshaped = sim_map.view(batch_size, query_H, query_W, key_H, key_W)
    sim_map_reshaped = sim_map.view(batch_size, query_H * query_W, key_H, key_W)
    
    # 对query维度取平均，得到每个key位置的注意力权重 (B, key_H, key_W)
    # att_map = sim_map_reshaped.mean(dim=(1, 2))  # 平均所有query位置的注意力
    # att_map = sim_map_reshaped.max(dim=(1, 2))[0]

    # att_map = sim_map_reshaped[:,8,8,:,:]  # 取最大响应作为注意力
    att_map = sim_map_reshaped.min(dim=1)[0]  # 取所有query位置的最大响应作为注意力
    print(att_map.shape)

    # 上采样到原图尺寸
    # att_map_up = F.interpolate(
    #     att_map.unsqueeze(1),  # 增加通道维度 (B, 1, key_H, key_W)
    #     size=[512, 512],  # 原图尺寸
    #     mode='bilinear',
    #     align_corners=False
    # ).squeeze(1)  # (B, orig_H, orig_W)
    att_map_up = att_map.squeeze(1)  # (B, key_H, key_W)

    # 提取单个样本的注意力图（取第0个样本）
    att_np = att_map_up[0].detach().cpu().numpy()

    #######################################
    # 关键优化：值范围拉伸与对比度增强
    #######################################
    # 1. 百分位截断：过滤极端小值（保留95%的有效范围，可根据需求调整）
    # 例如：截断低于5%分位的值，高于95%分位的值按95%分位处理
    p_low = np.percentile(att_np, 2)   # 第0百分位（排除过小的噪声值）
    p_high = np.percentile(att_np, 98) # 第100百分位（排除过大的异常值）
    att_truncated = np.clip(att_np, p_low, p_high)  # 截断超出范围的值

    # 2. 归一化到0-1（基于截断后的范围）
    att_norm = (att_truncated - p_low) / (p_high - p_low + 1e-8)

    # 3. 非线性拉伸：增强低数值区域的对比度（可选平方根或对数变换）
    # 平方根变换：对低数值更敏感，适合值分布集中在低区间的情况
    att_stretched = np.sqrt(att_norm)  # 若效果仍不足，可尝试 att_stretched = att_norm ** 0.3
    # 若数据有明显偏态，也可尝试对数变换（需保证值为正，这里满足）：
    # att_stretched = np.log(att_norm + 1e-6)  # +1e-6避免log(0)
    # att_stretched = (att_stretched - att_stretched.min()) / (att_stretched.max() - att_stretched.min() + 1e-8)  # 再归一化

    # 4. 映射到0-255并转换为uint8
    att_scaled = (att_stretched * 255).astype(np.uint8)

    # 转换为热力图（JET配色）
    heatmap = cv2.applyColorMap(att_scaled, cv2.COLORMAP_JET)
    
    # 保存路径
    save_path = os.path.join(
        "./result/attention_maps/",
        f"{index}_{suffix}.png"
    )
    cv2.imwrite(save_path, heatmap)
    print(f"优化后的注意力图已保存至: {save_path}")

def save_feature_map(feat, index, suffix):
    """保存特征图为热力图"""

    # 计算每个空间位置的特征强度（L2范数）
    # feature = feat.detach().cpu().numpy()
    feature = F.interpolate(
        feat,
        size=[512, 512],  # 原图尺寸
        mode='bilinear',
        align_corners=False
    ).detach().cpu().numpy()
    print(feature.shape)
    # img_out = np.mean(feature[0], axis=0)

    # 是否减去极端值更好？

    img_out = feature[0][1]
    cv2.normalize(img_out, img_out, 0, 255, cv2.NORM_MINMAX)
    img_out = np.array(img_out, dtype=np.uint8)
    img_out = cv2.applyColorMap(img_out, cv2.COLORMAP_JET)
    save_path = os.path.join(
        "./result/attension_maps_fbps/",
        f"{index}_{suffix}.png"
    )

    cv2.imwrite(save_path, img_out)

def save_pca_map(feat, index, suffix):
    """
    处理token特征：直接将256个token重排为16×16网格，再进行PCA

    参数:
        feat: 输入特征张量，形状为 [B, C, 16, 16],这里C=1536 ，B=1
    返回:
        pca_map: PCA降维后的32x32语义图（3通道用于可视化）
    """
    # 移除batch维度，得到 [256, 1024]（256个token，每个1024维）

    print(feat.shape)
    feature = F.interpolate(
        feat,
        size=[512, 512],  # 原图尺寸
        mode='bilinear',
        align_corners=False
    )
    tokens = feature.squeeze(0).permute(1, 2, 0)  # 形状: [256, 1024]

    # 直接将256个token重排为16×16的网格（16×16=256）
    # 每个网格位置对应一个原始token，保留其1024维特征
    # token_grid = tokens.reshape(16, 16, 1024)  # 形状: [16, 16, 1024]
    token_grid = tokens.detach().cpu().numpy()  # 形状: [32, 32, 1024]

    # 准备PCA输入：将32×32的网格展平为1024个样本，每个样本1024维
    pca_input = token_grid.reshape(-1, 1536)  # 形状: [512*512, 1536]

    # PCA降维：将1024维特征压缩到3维（对应RGB通道）
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(pca_input)  # 形状: [1024, 3]

    # 归一化到[0,1]范围，适合图像显示
    scaler = MinMaxScaler()
    pca_features = scaler.fit_transform(pca_features)  # 形状: [1024, 3]

    # 重排回32×32网格，得到语义图
    pca_map = pca_features.reshape(512, 512, 3)  # 形状: [32, 32, 3]

    cv2.imwrite(f'./result/pca_map/{index}_{suffix}.png', pca_map[:,:,::-1]*255)  # 保存为PNG图像

@MODELS.register_module()
class CascadedCrossScaleFusion(nn.Module):
    """Cascaded Cross-Scale Fusion Neck.

    A neck structure connect multi-level feature backbone and decoder_heads.
    """

    def __init__(self,
                 main_channels=1536,
                 sub_channels=[512,512],
                 ab_2 = False,
                 ab_3 = False,
                 vis_flag = False
                 ):
        super().__init__()

        self.mlp_align_1 = MLPAlign(sub_channels[0], main_channels//2, 3, act_cfg=nn.ReLU())
        self.mlp_align_2 = MLPAlign(sub_channels[1], main_channels//2, 3, act_cfg=nn.ReLU())

        self.ccsf1 = LowRankAttention(channels=main_channels)
        self.ccsf2 = LowRankAttention(channels=main_channels)

        self.ab_2 = ab_2
        self.ab_3 = ab_3

        self.vis_flag = vis_flag
        self.save_id = 0

    def forward(self, inputs):
        
        outputs = []
        outputs.append(inputs[0]) # the lowest level feature, no CCSF
        outputs.append(inputs[1]) # the second level feature, no CCSF
        outputs.append(inputs[2]) # the third level feature, no CCSF

        # from local to global cfi
        query = inputs[3].clone() # main encoder feature 3
        fusion_out = inputs[3].clone()

        # save_pca_map(query, self.save_id, "main_encoder_feat")
        # save_feature_map(fusion_out, self.save_id, "main_encoder_feat")

        # for ab_2
       
        query, sim_map_short = self.ccsf1(query, self.mlp_align_1(inputs[4]))
        fusion_out += query # sub encoder feature short range

        # save_pca_map(query, self.save_id, "short_range_fusion_out")
        # save_feature_map(fusion_out, self.save_id, "short_range_fusion_out")

        # for ab_3
        query, sim_map_long = self.ccsf2(query, self.mlp_align_2(inputs[5]))
        fusion_out += query # sub encoder feature long range

        # save_pca_map(query, self.save_id, "long_range_fusion_out")
        # save_feature_map(fusion_out, self.save_id, "long_range_fusion_out")

        self.save_id += 1

        # self.vis_flag = False  # set to True to save attention maps
        # if self.vis_flag: 
        #     save_attention_heatmap(sim_map_short, self.save_id, "short_range_att")
        #     save_attention_heatmap(sim_map_long, self.save_id, "long_range_att")
        #     self.save_id += 1

        outputs.append(fusion_out)
        outputs.append(inputs[3]) # for auxiliary head

        return tuple(outputs)



