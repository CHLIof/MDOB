# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.structures.det_data_sample import SampleList
from mmdet.utils import InstanceList, OptConfigType

from mmcv.cnn import  build_norm_layer, build_activation_layer

from mmdet.models.layers.transformer.CrossAttention import ScaledDotProductAttention


@MODELS.register_module()
class EmbeddingRPNHeadDetectionHub(BaseModule):
    """RPNHead in the `Sparse R-CNN <https://arxiv.org/abs/2011.12450>`_ .

    Unlike traditional RPNHead, this module does not need FPN input, but just
    decode `init_proposal_bboxes` and expand the first dimension of
    `init_proposal_bboxes` and `init_proposal_features` to the batch_size.

    Args:
        num_proposals (int): Number of init_proposals. Defaults to 100.
        proposal_feature_channel (int): Channel number of
            init_proposal_feature. Defaults to 256.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 num_proposals: int = 100,
                 proposal_feature_channel: int = 256,
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        # `**kwargs` is necessary to avoid some potential error.
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)
        self.num_proposals = num_proposals
        self.proposal_feature_channel = proposal_feature_channel
        self._init_layers()

        self.cross_attention = ScaledDotProductAttention(d_model=256, d_k=256, d_v=256, h=8)

        self.dynamic_layer = nn.Linear(
            self.num_proposals * self.proposal_feature_channel , self.num_proposals * self.num_proposals)

        self.dot_product_projection_text = nn.Linear(
            768,
            256,
            bias=True)

        self.norm = build_norm_layer({'type': 'LN'}, 4)[1]
        self.activation = build_activation_layer({'type':'ReLU', 'inplace':True})

    def _init_layers(self) -> None:
        """Initialize a sparse set of proposal boxes and proposal features."""
        self.init_proposal_bboxes = nn.Embedding(self.num_proposals, 4)
        self.init_proposal_features = nn.Embedding(
            self.num_proposals, self.proposal_feature_channel)

    def init_weights(self) -> None:
        """Initialize the init_proposal_bboxes as normalized.

        [c_x, c_y, w, h], and we initialize it to the size of  the entire
        image.
        """
        super().init_weights()
        nn.init.constant_(self.init_proposal_bboxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_bboxes.weight[:, 2:], 1)

    def _decode_init_proposals(self, x: List[Tensor],
                               batch_data_samples: SampleList, language_dict_features=None) -> InstanceList:
        """Decode init_proposal_bboxes according to the size of images and
        expand dimension of init_proposal_features to batch_size.

        Args:
            x (list[Tensor]): List of FPN features.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            List[:obj:`InstanceData`:] Detection results of each image.
            Each item usually contains following keys.

            - proposals: Decoded proposal bboxes,
              has shape (num_proposals, 4).
            - features: init_proposal_features, expanded proposal
              features, has shape
              (num_proposals, proposal_feature_channel).
            - imgs_whwh: Tensor with shape
              (num_proposals, 4), the dimension means
              [img_width, img_height, img_width, img_height].
        """

        # print(self.init_proposal_bboxes.weight)
        # print(self.init_proposal_features.weight)
        # exit(0)

        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)

        proposals = self.init_proposal_bboxes.weight.clone()
        # proposals = self.init_proposal_bboxes.weight
        print(proposals)

        # exit(0)
        proposals = bbox_cxcywh_to_xyxy(proposals)
        imgs_whwh = []
        for meta in batch_img_metas:
            h, w = meta['img_shape'][:2]
            imgs_whwh.append(x[0].new_tensor([[w, h, w, h]]))

        print(imgs_whwh)
        # exit(0)

        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]
        print(imgs_whwh)
        # exit(0)

        proposals = proposals * imgs_whwh

        # # print(proposals.size())
        # # print(language_dict_features['embedded'].size())

        # embedding = language_dict_features['embedded']
        # embedding = F.normalize(embedding, p=2, dim=-1)
        # language_dict_features['embedding'] = self.dot_product_projection_text(embedding /
        #                                                                       2.0)
        # #
        proposal_feat=[]
        for idx in range(len(batch_img_metas)):
            proposal_feat.append(self.init_proposal_features.weight.clone().unsqueeze(0))

        proposal_feat = torch.cat(proposal_feat, dim=0)
        #
        # # proposal_feat = self.init_proposal_features.weight.clone()
        #
        # # print(proposal_feat.size())
        # # exit(0)
        # embedding = language_dict_features['embedded']
        # # print(embedding.size())
        # embedding = self.dot_product_projection_text(embedding)
        #
        # # exit(0)
        #
        #
        # # proposal_feat = self.init_proposal_features.weight.clone()
        dot_product_proj_tokens = language_dict_features['embedding']
        # # print(dot_product_proj_tokens.size())
        # # # print(proposal_feat.size())
        # # exit(0)
        # # proposal_feat = proposal_feat.unsqueeze(0)
        #
        #
        proposal_feat = self.cross_attention(proposal_feat, dot_product_proj_tokens, dot_product_proj_tokens)
        # proposal_feat = proposal_feat.squeeze(0)
        # print(proposal_feat.size())
        # print(proposals.size())
        # # #
        proposal_feat_adapt_query = proposal_feat.view(proposal_feat.size(0), -1)
        # print(proposal_feat_adapt_query.size())
        parameters = self.dynamic_layer(proposal_feat_adapt_query)
        # print(parameters.size())


        bs, channel, h = proposals.size()

        parameters = parameters.view(bs * self.num_proposals, -1, 1, 1)

        # print(parameters.size())
        # exit(0)

        proposals_tensor = proposals.view(-1, bs * channel, h, 1)

        # print(proposals_tensor.size())

        proposals_tensor = F.conv2d(proposals_tensor, weight=parameters, bias=None, stride=1, padding=0, dilation=1, groups=1 * bs)
        # print(proposals_tensor.size())
        # #
        # proposals_tensor = proposals_tensor.squeeze(-1)
        # #
        proposals_tensor = proposals_tensor.view(bs, channel, -1)

        proposals_tensor = self.norm(proposals_tensor)
        proposals_tensor = self.activation(proposals_tensor)
        # print(proposals_tensor.size())
        # print(proposals_tensor.size())


        # exit(0)
        # # # proposal_feat_adapt_query = proposal_feat.permute(1, 0, 2)
        # # proposal_feat_adapt_query = proposal_feat_adapt_query.reshape(self.num_proposals, -1)
        #
        # # print(proposal_feat_adapt_query.size())
        # if proposal_feat.size(0) > 1:
        #     proposal_feat = proposal_feat.mean(dim=0)
        #
        #     # print(proposal_feat.size())
        #     # exit(0)
        #
        # parameters = self.dynamic_layer(proposal_feat)
        # #
        # # # print(parameters.size())
        # parameters = parameters.view(self.num_proposals, self.num_proposals, 1, 1)
        # #
        # proposals = proposals.unsqueeze(3)
        # # # print(proposals.size())
        # #
        # proposals = F.conv2d(proposals, weight=parameters, bias=None, stride=1, padding=0, dilation=1, groups=1)
        # #
        # proposals = proposals.squeeze(3)
        # # print(proposals.size())
        # # exit(0)
        # proposals = self.norm(proposals)
        # # proposals = self.activation(proposals)
        # # print(proposals.size())
        # # exit(0)

        rpn_results_list = []
        for idx in range(len(batch_img_metas)):
            rpn_results = InstanceData()
            rpn_results.bboxes = proposals_tensor[idx]
            rpn_results.imgs_whwh = imgs_whwh[idx].repeat(
                self.num_proposals, 1)
            rpn_results.features = proposal_feat[idx]
            rpn_results_list.append(rpn_results)
        return rpn_results_list

    def loss(self, *args, **kwargs):
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network."""
        raise NotImplementedError(
            'EmbeddingRPNHead does not have `loss`, please use '
            '`predict` or `loss_and_predict` instead.')

    def predict(self, x: List[Tensor], batch_data_samples: SampleList, language_dict_features,
                **kwargs) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network."""
        # `**kwargs` is necessary to avoid some potential error.
        return self._decode_init_proposals(
            x=x, batch_data_samples=batch_data_samples, language_dict_features=language_dict_features)

    def loss_and_predict(self, x: List[Tensor], batch_data_samples: SampleList, language_dict_features,
                         **kwargs) -> tuple:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples."""
        # `**kwargs` is necessary to avoid some potential error.
        predictions = self._decode_init_proposals(
            x=x, batch_data_samples=batch_data_samples, language_dict_features=language_dict_features)

        return dict(), predictions
