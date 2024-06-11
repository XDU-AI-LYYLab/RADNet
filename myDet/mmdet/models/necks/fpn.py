import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule
import torch
from ..plugins import Attention2D
#------------------------------------heatmap---------------------------
from ..heatmap import visualize_feature_map,draw_features
# heatmappath = '/home/lyy/hq/DTestRes/paper2/att'

@NECKS.register_module
class FPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 attention=False,
                 upsample_mode = 'nearest'):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
#--------------------------------------------------de convolution -------------------------------

        self.upsample_mode = upsample_mode
        self.upsample_block_module = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)


        # =========================== add attention ==========================
        self.attention = attention

        if self.attention:
            #=================== attention ==============================
            self.att_block = Attention2D(in_channels=self.out_channels, pooling_type='simple_att',fusion_types=('conv_add', ))

        # ======================================================================
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if self.upsample_mode == 'nearest':
                temp = F.interpolate(
                    laterals[i], scale_factor=2, mode='nearest')
            # -----------------------------------------draw_features--------------------------------
            # if i== used_backbone_levels - 1:
            #     b, c, h, w = temp.shape
            #     draw_features(c, w, h, temp.cpu().numpy(), savename='fpn_temp', savepath=heatmappath+'/fpn')
            #--------------------------------------------------------------------------------------

            if self.upsample_mode == 'deconv':
                temp = self.upsample_block_module(laterals[i])
            laterals[i - 1] += temp

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels) #P2,P3,P4,P5
        ]

        #---------------------------draw feature map ---------
        # visualize_feature_map(outs[0].cpu().numpy(), savename='before_att', savepath=heatmappath)
        # b, c, h, w = outs[0].shape
        # draw_features(c, w, h, outs[0].cpu().numpy(), savename='before_att', savepath=heatmappath)

        # =========================== add attention ==========================
        if self.attention:
            out_temp = []
            for i,out in enumerate(outs): # out[0]先下采样，通过注意力后，上采样
                out_tmp = out
                final = self.att_block(out_tmp)
                out_temp.append(final)
            outs = out_temp

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
