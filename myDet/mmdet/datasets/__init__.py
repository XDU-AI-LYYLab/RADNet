from .builder import build_dataset
from .cityscapes import CityscapesDataset
from .coco import CocoDataset
from .custom import CustomDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .loader import DistributedGroupSampler, GroupSampler, build_dataloader
from .registry import DATASETS
from .voc import VOCDataset
from .wider_face import WIDERFaceDataset
from .xml_style import XMLDataset
#-----------------新添加的数据集------------
from .dota import DotaDataset
from .dior import DiorDataset
from .sdd import SddDataset
from .sdota import SdotaDataset
from .NWPUVHR_10 import NWPUVHR_Dataset
from .OHD import OhdDataset
from .MASATI import MasatiDataset
from .Carpk import CarpkDataset

__all__ = [
    'CustomDataset', 'XMLDataset', 'CocoDataset', 'VOCDataset',
    'CityscapesDataset', 'GroupSampler', 'DistributedGroupSampler',
    'build_dataloader', 'ConcatDataset', 'RepeatDataset', 'WIDERFaceDataset',
    'DATASETS', 'build_dataset',
    "DotaDataset", "DiorDataset", "SddDataset", "SdotaDataset","NWPUVHR_Dataset" ,"OhdDataset", "MasatiDataset", "CarpkDataset"#----------------新添加的数据集
]
