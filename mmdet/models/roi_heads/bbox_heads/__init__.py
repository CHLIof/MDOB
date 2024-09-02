# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_head import BBoxHead
from .convfc_bbox_head import (ConvFCBBoxHead, Shared2FCBBoxHead,
                               Shared4Conv1FCBBoxHead)
from .dii_head import DIIHead
from .double_bbox_head import DoubleConvFCBBoxHead
from .multi_instance_bbox_head import MultiInstanceBBoxHead
from .sabl_head import SABLHead
from .scnet_bbox_head import SCNetBBoxHead
from .convfc_bbox_head_DA import Shared2FCBBoxHeadDA

from .convfc_bbox_head_scaleDet import Shared2FCBBoxHeadScaleDet
from .convfc_bbox_head_scaleDet_similarity import Shared2FCBBoxHeadScaleDetSimilarity

from .dii_head_language import DIIHeadLanguage
from .dii_head_language_DetectionHub import DIIHeadLanguageDetectionHub

__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead',
    'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead', 'SABLHead', 'DIIHead',
    'SCNetBBoxHead', 'MultiInstanceBBoxHead', 'Shared2FCBBoxHeadDA', 'Shared2FCBBoxHeadScaleDet', 'Shared2FCBBoxHeadScaleDetSimilarity',
    'DIIHeadLanguage', 'DIIHeadLanguageDetectionHub'
]
