# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import (cat_boxes, empty_box_as, get_box_tensor,
                                   get_box_wh, scale_boxes)
from mmdet.utils import InstanceList, MultiConfig, OptInstanceList
from .anchor_head import AnchorHead

from mmdet.structures import SampleList

from ..utils import (unpack_gt_instances, multi_apply)

from mmdet.models.layers.transformer.CrossAttention import ScaledDotProductAttention



class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)




@MODELS.register_module()
class RPNHeadDetectionHub(AnchorHead):
    """Implementation of RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of categories excluding the background
            category. Defaults to 1.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or \
            list[dict]): Initialization config dict.
        num_convs (int): Number of convolution layers in the head.
            Defaults to 1.
    """  # noqa: W605

    def __init__(self,
                 in_channels: int,
                 num_classes: int = 1,
                 init_cfg: MultiConfig = dict(
                     type='Normal', layer='Conv2d', std=0.01),
                 num_convs: int = 1,
                 num_proposals: int=100,
                 **kwargs) -> None:
        self.num_convs = num_convs
        assert num_classes == 1

        self.num_proposals = num_proposals

        self.proposal_feature_channel = 256

        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                # use ``inplace=False`` to avoid error: one of the variables
                # needed for gradient computation has been modified by an
                # inplace operation.
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        inplace=False))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(
                self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * self.cls_out_channels,
                                 1)
        reg_dim = self.bbox_coder.encode_size
        self.rpn_reg = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * reg_dim, 1)

        self.init_proposal_features = nn.Embedding(
            self.num_proposals, self.proposal_feature_channel)

        self.cross_attention = ScaledDotProductAttention(d_model=256, d_k=256, d_v=256, h=8)

        self.dynamic_layer = nn.Linear(
            self.num_proposals , self.proposal_feature_channel *3*3)

        self.K = 4
        self.kernel_size = 3
        # self.attention = attention2d(self.num_proposals * self.proposal_feature_channel, 0.25, self.K, 34)

        # self.weight = nn.Parameter(torch.randn(self.K, self.proposal_feature_channel, self.proposal_feature_channel, 3, 3),
        #                            requires_grad=True)



    def forward(self, x: Tuple[Tensor], parameters) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        return multi_apply(self.forward_single, x, parameters)

    def forward_single(self, x: Tensor, parameters) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_base_priors * 4.
        """

        # print(parameters.size())
        # exit(0)

        # print(x.size())
        # exit(0)

        x = self.rpn_conv(x)


        # query-based ___start
        # bs = x.size(0)
        # x = x.view(1, -1, x.size(2), x.size(3))
        # x = F.conv2d(x, weight=parameters, bias=None, stride=1, padding=1,
        #                       dilation=1, groups=1*bs)
        # # print(x.size())
        # x = x.view(bs, -1, x.size(2), x.size(3))

        # query-based ___end
        # exit(0)

        x = F.relu(x)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred


    def loss_and_predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        language_dict_features,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.
            proposal_cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.

        Returns:
            tuple: the return value is a tuple contains:

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - predictions (list[:obj:`InstanceData`]): Detection
                  results of each image after the post process.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        bs = len(batch_img_metas)
        proposal_feat = []
        for idx in range(len(batch_img_metas)):
            proposal_feat.append(self.init_proposal_features.weight.clone().unsqueeze(0))

        proposal_feat = torch.cat(proposal_feat, dim=0)

        dot_product_proj_tokens = language_dict_features['embedding']

        proposal_feat = self.cross_attention(proposal_feat, dot_product_proj_tokens, dot_product_proj_tokens)

        # print(proposal_feat.size())
        # exit(0)

        proposal_feat_adapt_query = proposal_feat.permute(0, 2, 1)

        # print(proposal_feat_adapt_query.size())
        # exit(0)

        # softmax_attention = self.attention(proposal_feat_adapt_query)
        # weight = self.weight.view(self.K, -1)
        # parameters = torch.mm(softmax_attention, weight).view(bs*self.proposal_feature_channel, self.proposal_feature_channel, self.kernel_size, self.kernel_size)

        parameters = self.dynamic_layer(proposal_feat_adapt_query)
        # print(parameters.size())

        parameters = parameters.view(bs* self.proposal_feature_channel, -1, self.kernel_size, self.kernel_size)

        # print(parameters.size())
        #
        # exit(0)

        # exit(0)
        parameters_list = []
        for i in x:
            parameters_list.append(parameters)

        outs = self(x, parameters_list)

        # print(outs[0].size())
        # print(outs[1].size())
        # print(outs[2].size())
        # print(outs[3].size())
        # exit(0)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)

        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, cfg=proposal_cfg)

        # print(predictions)

        # exit(0)

        predictions2 = []
        for ind, data_sample in enumerate(batch_data_samples):
            # print(data_sample)

            img_shape = data_sample.img_shape

            # print(img_shape)

            h, w = img_shape[:2]

            imgs_whwh = x[0].new_tensor([[w, h, w, h]])

            imgs_whwh = imgs_whwh.repeat(
                self.num_proposals, 1)

            # print(imgs_whwh)

            results = InstanceData()

            bboxes = predictions[ind].bboxes[:self.num_proposals]
            scores = predictions[ind].scores[:self.num_proposals]
            labels = predictions[ind].labels[:self.num_proposals]


            # print(bboxes)
            #
            # print(scores)
            # print(scores.size())

            # topk = torch.topk(scores, self.num_proposals)
            #
            # print(topk)
            results.bboxes = bboxes
            results.scores = scores
            results.labels = labels
            results.imgs_whwh = imgs_whwh
            results.features = proposal_feat[ind]

            predictions2.append(results)


            # exit(0)

        # print(predictions2)
        # exit(0)

        # return losses, predictions
        return losses, predictions2


    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                language_dict_features,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        bs = len(batch_img_metas)
        proposal_feat = []
        for idx in range(len(batch_img_metas)):
            proposal_feat.append(self.init_proposal_features.weight.clone().unsqueeze(0))

        proposal_feat = torch.cat(proposal_feat, dim=0)

        dot_product_proj_tokens = language_dict_features['embedding']

        proposal_feat = self.cross_attention(proposal_feat, dot_product_proj_tokens, dot_product_proj_tokens)

        # proposal_feat_adapt_query = proposal_feat.view(bs, -1, 1, 1)
        #
        # softmax_attention = self.attention(proposal_feat_adapt_query)
        # weight = self.weight.view(self.K, -1)
        # parameters = torch.mm(softmax_attention, weight).view(bs * self.proposal_feature_channel,
        #                                                       self.proposal_feature_channel, self.kernel_size,
        #                                                       self.kernel_size)

        proposal_feat_adapt_query = proposal_feat.permute(0, 2, 1)

        parameters = self.dynamic_layer(proposal_feat_adapt_query)

        parameters = parameters.view(bs * self.proposal_feature_channel, -1, self.kernel_size, self.kernel_size)

        parameters_list = []
        for i in x:
            parameters_list.append(parameters)

        outs = self(x, parameters_list)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)

        predictions2 = []
        for ind, data_sample in enumerate(batch_data_samples):

            img_shape = data_sample.img_shape

            h, w = img_shape[:2]

            imgs_whwh = x[0].new_tensor([[w, h, w, h]])

            imgs_whwh = imgs_whwh.repeat(
                self.num_proposals, 1)

            results = InstanceData()

            bboxes = predictions[ind].bboxes[:self.num_proposals]
            scores = predictions[ind].scores[:self.num_proposals]
            labels = predictions[ind].labels[:self.num_proposals]

            results.bboxes = bboxes
            results.scores = scores
            results.labels = labels
            results.imgs_whwh = imgs_whwh
            results.features = proposal_feat[ind]

            predictions2.append(results)

        # return predictions
        return predictions2


    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) \
            -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[obj:InstanceData]): Batch of gt_instance.
                It usually includes ``bboxes`` and ``labels`` attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[obj:InstanceData], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = super().loss_by_feat(
            cls_scores,
            bbox_preds,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Be compatible with
                BaseDenseHead. Not used in RPNHead.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (ConfigDict, optional): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        level_ids = []
        for level_idx, (cls_score, bbox_pred, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            reg_dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, reg_dim)
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0] since mmdet v2.0
                # BG cat_id: 1
                scores = cls_score.softmax(-1)[:, :-1]

            scores = torch.squeeze(scores)
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                bbox_pred = bbox_pred[topk_inds, :]
                priors = priors[topk_inds]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)

            # use level id to implement the separate level nms
            level_ids.append(
                scores.new_full((scores.size(0), ),
                                level_idx,
                                dtype=torch.long))

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.level_ids = torch.cat(level_ids)

        return self._bbox_post_process(
            results=results, cfg=cfg, rescale=rescale, img_meta=img_meta)

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert with_nms, '`with_nms` must be True in RPNHead'
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        if results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                results.level_ids, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]
            # TODO: This would unreasonably show the 0th class label
            #  in visualization
            results.labels = results.scores.new_zeros(
                len(results), dtype=torch.long)
            del results.level_ids
        else:
            # To avoid some potential error
            results_ = InstanceData()
            results_.bboxes = empty_box_as(results.bboxes)
            results_.scores = results.scores.new_zeros(0)
            results_.labels = results.scores.new_zeros(0)
            results = results_
        return results
