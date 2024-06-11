import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule, ConvModule2, ConvModule3
from ..plugins import Attention2D

from ..heatmap import visualize_feature_map, draw_features

heatmappath = '/home/lyy/hq/DTestRes/miniDOTA/heatmap'


@NECKS.register_module
class RFPN(nn.Module):

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
                 upsample_mode='resize_conv'):
        super(RFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.fp16_enabled = False
        self.upsample_mode = upsample_mode
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
        # =========================== add attention ==========================
        self.attention = attention

        if self.attention:
            # =================== self attention ==============================
            self.att_block = Attention2D(in_channels=self.out_channels, pooling_type='simple_att',fusion_types=('conv_add',))

            # ================== channel attention =============================
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(out_channels, out_channels // 16, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels // 16, out_channels, bias=False),
                nn.Sigmoid())

        # ======================================================================
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.add_convs = nn.ModuleList()
        # **********************************************
        self.stride_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        # *************************************************
        # --------------------resize conv的 3*3卷积 ---------------------------------------
        self.conv3_3 = ConvModule(
            out_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=self.activation,
            inplace=False)
        # ====================== 转置卷积 ===========================
        self.upsample_block_module = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        # =========================================================
        for i in range(self.start_level, self.backbone_end_level):
            # ------------------- rfpn 横向连接 3*3 relu 3*3-----------------------------
            l_conv = ConvModule2(
                in_channels[i],
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation="relu",
                inplace=False)

            # ------------------------------// relu 3*3 relu //--------------------
            fpn_conv = ConvModule3(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation="relu",
                inplace=False)


            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        # print(self.lateral_convs)
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
        # build laterals   STEP 1
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build top-down path  STEP 2
        outs = []
        used_backbone_levels = len(laterals)  # 4
        for i in range(used_backbone_levels - 1, 0, -1):
            # ========== resize conv=======================
            if self.upsample_mode == 'resize_conv':
                temp = F.interpolate(laterals[i], scale_factor=2, mode='nearest')
                conv_temp = self.conv3_3(temp)
            # ===========deconv=========================
            if self.upsample_mode == 'deconv':
                conv_temp = self.upsample_block_module(laterals[i])
            laterals[i - 1] += conv_temp

        # build outputs  STEP 3

        # part 1: from original levels
        # --------------------modify-------------3.28
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)  # P2,P3,P4,P5
        ]
        # ---------跳连接 -----------------------2020.12.17
        for i, l in enumerate(laterals):
            # print(l.shape)
            outs[i] += l

        # =========================== add attention ==========================
        if self.attention:
            # print("-----use attention-----")
            out_temp = []
            for i, out in enumerate(outs):  # out[0]先下采样，通过注意力后，上采样
                out_tmp = out
                # ======================= GC attention=============================
                final = self.att_block(out_tmp)

                out_temp.append(final)
            outs = out_temp
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))  # P6层 # paouts
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
