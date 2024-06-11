import torch
import torch.nn as nn
from mmcv.cnn import constant_init, normal_init
import torch.nn.functional as F
from ..utils import ConvModule

import torch

from mmcv.cnn import constant_init, kaiming_init

from torch import nn

from ..heatmap import visualize_feature_map,draw_features
heatmappath = '/home/lyy/hq/DTestRes/maskRCNN/DOTA/heatmap'
def last_zero_init(m):

    if isinstance(m, nn.Sequential):

        constant_init(m[-1], val=0)

    else:

        constant_init(m, val=0)

class Attention2D(nn.Module):
    """Non-local module.

    See https://arxiv.org/abs/1711.07971 for details.

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio.
        use_scale (bool): Whether to scale pairwise_weight by 1/inter_channels.
        conv_cfg (dict): The config dict for convolution layers.
            (only applicable to conv_out)
        norm_cfg (dict): The config dict for normalization layers.
            (only applicable to conv_out)
        mode (str): Options are `embedded_gaussian` and `dot_product`.
    """

    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian',
                 ratio=1./4,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(Attention2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product']
#-----------------------------------------GC -----------------------
        assert pooling_type in ['avg', 'att','simple_att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul','conv_add']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.ratio = ratio
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        self.planes = int(in_channels * ratio)
        print("pooling_type：",pooling_type)
        print("fusion_types：",fusion_types)
        if pooling_type == 'simple_att': #------------------------------------------add
            self.conv_mask = nn.Conv2d(self.in_channels, 1, kernel_size=1)
            self.v = ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=1,
                    activation=None)
            self.softmax = nn.Softmax(dim=2)
        elif pooling_type == 'att':

            self.conv_mask = nn.Conv2d(self.in_channels, 1, kernel_size=1)

            self.softmax = nn.Softmax(dim=2)

        else:

            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if 'channel_add' in fusion_types:

            self.channel_add_conv = nn.Sequential(

                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),

                nn.LayerNorm([self.planes, 1, 1]),

                nn.ReLU(inplace=True),  # yapf: disable

                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
            # -------------add----------
        elif 'conv_add' in fusion_types:
            self.channel_add_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        else:

            self.channel_add_conv = None

        if 'channel_mul' in fusion_types:

            self.channel_mul_conv = nn.Sequential(

                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),

                nn.LayerNorm([self.planes, 1, 1]),

                nn.ReLU(inplace=True),  # yapf: disable

                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
        #-------------add----------
        elif 'conv_mul' in fusion_types:
            self.channel_mul_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        else:

            self.channel_mul_conv = None

        self.reset_parameters()

#----------------------------------------------------------------------------------
    def reset_parameters(self):

        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')

            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)

        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):

        batch, channel, height, width = x.shape

        if self.pooling_type == 'simple_att': #----------------add
            value = self.v(x)
            value = value.view(batch, channel, height * width)  # [N, C, H * W]
            value = value.unsqueeze(1)  # [N, 1, C, H * W]

            context_mask = self.conv_mask(x)   # [N, 1, H, W]
            context_mask = context_mask.view(batch, 1, height * width)   # [N, 1, H * W]
            context_mask = self.softmax(context_mask)     # [N, 1, H * W]
            context_mask = context_mask.unsqueeze(-1)     # [N, 1, H * W, 1]

            context = torch.matmul(value, context_mask)  # [N, 1, C, 1]
            context = context.view(batch, channel, 1, 1)   # [N, C, 1, 1]

        elif self.pooling_type == 'att':
            input_x = x  # [N, C, H, W]
            input_x = input_x.view(batch, channel, height * width)  # [N, C, H * W]
            input_x = input_x.unsqueeze(1)  # [N, 1, C, H * W]

            context_mask = self.conv_mask(x)  # [N, 1, H, W]
            context_mask = context_mask.view(batch, 1, height * width)  # [N, 1, H * W]
            context_mask = self.softmax(context_mask)  # [N, 1, H * W]
            context_mask = context_mask.unsqueeze(-1)  # [N, 1, H * W, 1]
            context = torch.matmul(input_x, context_mask)  # [N, 1, C, 1]
            context = context.view(batch, channel, 1, 1)  # [N, C, 1, 1]

        else:
            context = self.avg_pool(x) # [N, C, 1, 1]

        return context
#-----------------------------------------------------------------
    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]

            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))

            out = out * channel_mul_term

        if self.channel_add_conv is not None:
            # [N, C, 1, 1]

            channel_add_term = self.channel_add_conv(context)

            out = out + channel_add_term

        return out
