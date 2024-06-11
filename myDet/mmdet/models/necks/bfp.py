import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..plugins import NonLocal2D
from ..registry import NECKS
from ..utils import ConvModule
import torch
#------------------------------- 画特征图 -------------------
from ..heatmap import visualize_feature_map,draw_features
heatmappath = '/home/lyy/hq/DTestRes/paper2/bfp'

@NECKS.register_module
class BFP(nn.Module):
    """BFP (Balanced Feature Pyrmamids)

    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    https://arxiv.org/pdf/1904.02701.pdf for details.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.
        refine_type (str): Type of the refine op, currently support
            [None, 'conv', 'non_local'].
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=2,
                 refine_type=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(BFP, self).__init__()
        assert refine_type in [None, 'conv', 'non_local']

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level < self.num_levels

        if self.refine_type == 'conv':
            self.refine = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2D(
                self.in_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.num_levels

        # step 1: gather multi-level features by resize and average
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)
        # concate_feats = torch.cat(feats, dim=1)
        #
        # b, c, width, height = concate_feats.shape
        # y = self.avg_pool(concate_feats).view(b, c)
        # y = self.fc(y).view(b, c, 1, 1)
        #
        # final_c = out_tmp * y.expand_as(out_tmp)
        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        # outs.append(inputs[0])
        # outs.append(inputs[1])
        # outs.append(inputs[2])
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)

            # print(residual.shape)
            '''
            torch.Size([1, 256, 256, 256])
            torch.Size([1, 256, 128, 128])
            torch.Size([1, 256, 64, 64])
            torch.Size([1, 256, 32, 32])
            torch.Size([1, 256, 16, 16])
            '''

            outs.append(residual + inputs[i])

        # visualize_feature_map(outs[0].cpu().numpy(), savename='merged', savepath=heatmappath)
        # for la in outs:
        #     print(la.shape)
        return tuple(outs)
