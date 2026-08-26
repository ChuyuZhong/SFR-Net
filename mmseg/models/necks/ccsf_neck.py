
import torch.nn as nn
import torch
from mmengine.model.weight_init import constant_init
from torch.nn import functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS

class LowRankAttention(nn.Module):

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

        if self.out_project is not None:
            if not isinstance(self.out_project, ConvModule):
                constant_init(self.out_project, 0)

    def build_project(self, in_channels, channels, num_convs, use_conv_module,
                      conv_cfg, norm_cfg, act_cfg):

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

        return context

class MLPAlign(nn.Module):


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


@MODELS.register_module()
class CascadedCrossScaleFusion(nn.Module):

    def __init__(self,
                 main_channels=1536,
                 sub_channels=[512,512]
                 ):
        super().__init__()

        self.mlp_align_1 = MLPAlign(sub_channels[0], main_channels//2, 3, act_cfg=nn.ReLU())
        self.mlp_align_2 = MLPAlign(sub_channels[1], main_channels//2, 3, act_cfg=nn.ReLU())

        self.ccsf1 = LowRankAttention(channels=main_channels)
        self.ccsf2 = LowRankAttention(channels=main_channels)

    def forward(self, inputs):

        outputs = []
        outputs.append(inputs[0])
        outputs.append(inputs[1])
        outputs.append(inputs[2])
        query = inputs[3].clone()
        fusion_out = inputs[3].clone()

        query = self.ccsf1(query, self.mlp_align_1(inputs[4]))
        fusion_out += query

        query = self.ccsf2(query, self.mlp_align_2(inputs[5]))
        fusion_out += query

        outputs.append(fusion_out)
        outputs.append(inputs[3])

        return tuple(outputs)