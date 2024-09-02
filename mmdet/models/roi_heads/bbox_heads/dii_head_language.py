# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

try:
    from transformers import BertConfig
except ImportError:
    BertConfig = None

import math
from mmengine.structures import InstanceData

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.config import ConfigDict
from mmengine.model import bias_init_with_prob
from mmdet.utils import InstanceList

from torch import Tensor

import torch.nn.functional as F

from mmdet.models.losses import accuracy
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, reduce_mean
from .bbox_head import BBoxHead


MAX_CLAMP_VALUE = 50000


@MODELS.register_module()
class DIIHeadLanguage(BBoxHead):
    r"""Dynamic Instance Interactive Head for `Sparse R-CNN: End-to-End Object
    Detection with Learnable Proposals <https://arxiv.org/abs/2011.12450>`_

    Args:
        num_classes (int): Number of class in dataset.
            Defaults to 80.
        num_ffn_fcs (int): The number of fully-connected
            layers in FFNs. Defaults to 2.
        num_heads (int): The hidden dimension of FFNs.
            Defaults to 8.
        num_cls_fcs (int): The number of fully-connected
            layers in classification subnet. Defaults to 1.
        num_reg_fcs (int): The number of fully-connected
            layers in regression subnet. Defaults to 3.
        feedforward_channels (int): The hidden dimension
            of FFNs. Defaults to 2048
        in_channels (int): Hidden_channels of MultiheadAttention.
            Defaults to 256.
        dropout (float): Probability of drop the channel.
            Defaults to 0.0
        ffn_act_cfg (:obj:`ConfigDict` or dict): The activation config
            for FFNs.
        dynamic_conv_cfg (:obj:`ConfigDict` or dict): The convolution
            config for DynamicConv.
        loss_iou (:obj:`ConfigDict` or dict): The config for iou or
            giou loss.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 num_classes: int = 80,
                 num_ffn_fcs: int = 2,
                 num_heads: int = 8,
                 num_cls_fcs: int = 1,
                 num_reg_fcs: int = 3,
                 feedforward_channels: int = 2048,
                 in_channels: int = 256,
                 dropout: float = 0.0,
                 ffn_act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 dynamic_conv_cfg: ConfigType = dict(
                     type='DynamicConv',
                     in_channels=256,
                     feat_channels=64,
                     out_channels=256,
                     input_feat_shape=7,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN')),
                 loss_iou: ConfigType = dict(type='GIoULoss', loss_weight=2.0),
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(
            num_classes=num_classes,
            reg_decoded_bbox=True,
            reg_class_agnostic=True,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_iou = MODELS.build(loss_iou)
        self.in_channels = in_channels
        self.fp16_enabled = False
        self.attention = MultiheadAttention(in_channels, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.instance_interactive_conv = MODELS.build(dynamic_conv_cfg)
        self.instance_interactive_conv_dropout = nn.Dropout(dropout)
        self.instance_interactive_conv_norm = build_norm_layer(
            dict(type='LN'), in_channels)[1]

        self.ffn = FFN(
            in_channels,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.cls_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))

        # over load the self.fc_cls in BBoxHead
        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(in_channels, self.num_classes)
        else:
            self.fc_cls = nn.Linear(in_channels, self.num_classes + 1)

        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.reg_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.reg_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))
        # over load the self.fc_cls in BBoxHead
        self.fc_reg = nn.Linear(in_channels, 4)

        # language---start
        self.lang_cfg = BertConfig.from_pretrained(
            r"/home/ly/workspace/unifymodel/GLIP2/model/models--bert-base-uncased")
        bias_value = -math.log((1 - 0.01) / 0.01)

        self.lang_dim = self.lang_cfg.hidden_size

        self.dot_product_projection_text = nn.Linear(
            self.lang_dim,
            self.in_channels,
            bias=True)

        self.log_scale = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.bias_lang = nn.Parameter(
            torch.zeros(self.lang_dim), requires_grad=True)
        self.bias0 = nn.Parameter(
            torch.Tensor([bias_value]), requires_grad=True)

        # self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])

        # print(self.in_channels)
        # exit(0)

        assert self.reg_class_agnostic, 'DIIHead only ' \
            'suppport `reg_class_agnostic=True` '
        assert self.reg_decoded_bbox, 'DIIHead only ' \
            'suppport `reg_decoded_bbox=True`'

    def init_weights(self) -> None:
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        super().init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)




    def forward(self, roi_feat: Tensor, proposal_feat: Tensor, language_dict_features, label_positives) -> tuple:
        """Forward function of Dynamic Instance Interactive Head.

        Args:
            roi_feat (Tensor): Roi-pooling features with shape
                (batch_size*num_proposals, feature_dimensions,
                pooling_h , pooling_w).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size, num_proposals, feature_dimensions)

        Returns:
            tuple[Tensor]: Usually a tuple of classification scores
            and bbox prediction and a intermediate feature.

            - cls_scores (Tensor): Classification scores for
              all proposals, has shape
              (batch_size, num_proposals, num_classes).
            - bbox_preds (Tensor): Box energies / deltas for
              all proposals, has shape
              (batch_size, num_proposals, 4).
            - obj_feat (Tensor): Object feature before classification
              and regression subnet, has shape
              (batch_size, num_proposal, feature_dimensions).
            - attn_feats (Tensor): Intermediate feature.
        """

        # print(language_dict_features)
        # print('here')
        # exit(0)

        # print("label_positives: ", label_positives)
        # exit(0)


        N, num_proposals = proposal_feat.shape[:2]

        # Self attention
        proposal_feat = proposal_feat.permute(1, 0, 2)
        proposal_feat = self.attention_norm(self.attention(proposal_feat))
        attn_feats = proposal_feat.permute(1, 0, 2)

        # instance interactive
        proposal_feat = attn_feats.reshape(-1, self.in_channels)
        proposal_feat_iic = self.instance_interactive_conv(
            proposal_feat, roi_feat)
        proposal_feat = proposal_feat + self.instance_interactive_conv_dropout(
            proposal_feat_iic)
        obj_feat = self.instance_interactive_conv_norm(proposal_feat)

        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        cls_feat = obj_feat
        reg_feat = obj_feat

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        # cls_score = self.fc_cls(cls_feat).view(
        #     N, num_proposals, self.num_classes
        #     if self.loss_cls.use_sigmoid else self.num_classes + 1)

        # print(cls_feat.size())
        # # print(self.loss_cls.use_sigmoid)
        # print("cls_score: ", cls_score.size())
        # print("N: ", N)
        #
        # print(language_dict_features['embedded'].size())
        #
        # print("proposal_feat: ", proposal_feat.size())
        # print("roi_feat: ", roi_feat.size())
        #
        # print("num_imgs: ", num_imgs)

        dot_product_proj_queries = cls_feat.reshape(N, -1,  self.in_channels)
        # print(dot_product_proj_queries.size())

        embedding = language_dict_features['embedded']
        embedding = F.normalize(embedding, p=2, dim=-1)
        dot_product_proj_tokens = self.dot_product_projection_text(embedding /
                                                                   2.0)

        dot_product_proj_tokens_bias = torch.matmul(
            embedding, self.bias_lang) + self.bias0

        # print("dot_product_proj_tokens size: ", dot_product_proj_tokens.size())
        # print("dot_product_proj_tokens_bias: ", dot_product_proj_tokens_bias.size())

        bias = dot_product_proj_tokens_bias.unsqueeze(1)

        dot_product_logit = (torch.matmul(dot_product_proj_queries,
                            dot_product_proj_tokens.transpose(-1, -2)) /
                                    self.log_scale.exp()) + bias

        # print(dot_product_logit.size())

        dot_product_logit = torch.clamp(
            dot_product_logit, max=MAX_CLAMP_VALUE)
        dot_product_logit = torch.clamp(
            dot_product_logit, min=-MAX_CLAMP_VALUE)

        max_class_num = 0
        for dic in label_positives:
            class_num = len(dic)

            if class_num > max_class_num:
                max_class_num = class_num

        # print(max_class_num)
        # # exit(0)
        # print("dot_product_logit: ", dot_product_logit.size())

        # exit(0)

        combine_product_logit = torch.zeros((dot_product_logit.size(0),dot_product_logit.size(1), max_class_num)).to(dot_product_logit.device)

        # print("combine_product_logit: ", combine_product_logit.size())
        # # exit(0)
        # print(label_positives)
        # exit(0)

        for key, dic in enumerate(label_positives):

            for num in dic.keys():

                combine_product_logit[key, :, num-1] = dot_product_logit[key, :, dic[num]].mean(-1)

                # exit(0)

        # print(combine_product_logit)
        #
        # exit(0)


        cls_score = combine_product_logit

        # print(cls_score.size())
        #
        # exit(0)

        bbox_delta = self.fc_reg(reg_feat).view(N, num_proposals, 4)

        return cls_score, bbox_delta, obj_feat.view(
            N, num_proposals, self.in_channels), attn_feats

    def loss_and_target(self,
                        cls_score: Tensor,
                        bbox_pred: Tensor,
                        sampling_results: List[SamplingResult],
                        rcnn_train_cfg: ConfigType,
                        imgs_whwh: Tensor,
                        concat: bool = True,
                        reduction_override: str = None) -> dict:
        """Calculate the loss based on the features extracted by the DIIHead.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results, has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            imgs_whwh (Tensor): imgs_whwh (Tensor): Tensor with\
                shape (batch_size, num_proposals, 4), the last
                dimension means
                [img_width,img_height, img_width, img_height].
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None.

        Returns:
            dict: A dictionary of loss and targets components.
            The targets are only used for cascade rcnn.
        """

        # print(sampling_results)
        # print(cls_score.size())
        # exit(0)

        class_number = cls_score.size(1)

        cls_reg_targets = self.get_targets(
            sampling_results=sampling_results,
            rcnn_train_cfg=rcnn_train_cfg,
            class_number=class_number,
            concat=concat)
        (labels, label_weights, bbox_targets, bbox_weights) = cls_reg_targets

        losses = dict()
        # bg_class_ind = self.num_classes
        bg_class_ind = class_number
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos)


        # print(labels)
        # print(labels.size())
        # print(cls_score.size())

        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['pos_acc'] = accuracy(cls_score[pos_inds],
                                             labels[pos_inds])
        if bbox_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0),
                                                  4)[pos_inds.type(torch.bool)]
                imgs_whwh = imgs_whwh.reshape(bbox_pred.size(0),
                                              4)[pos_inds.type(torch.bool)]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred / imgs_whwh,
                    bbox_targets[pos_inds.type(torch.bool)] / imgs_whwh,
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
                losses['loss_iou'] = self.loss_iou(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                losses['loss_iou'] = bbox_pred.sum() * 0
        return dict(loss_bbox=losses, bbox_targets=cls_reg_targets)

    def _get_targets_single(self, pos_inds: Tensor, neg_inds: Tensor,
                            pos_priors: Tensor, neg_priors: Tensor,
                            pos_gt_bboxes: Tensor, pos_gt_labels: Tensor,
                            class_number,
                            cfg: ConfigDict) -> tuple:
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Almost the same as the implementation in `bbox_head`,
        we add pos_inds and neg_inds to select positive and
        negative samples instead of selecting the first num_pos
        as positive samples.

        Args:
            pos_inds (Tensor): The length is equal to the
                positive sample numbers contain all index
                of the positive sample in the origin proposal set.
            neg_inds (Tensor): The length is equal to the
                negative sample numbers contain all index
                of the negative sample in the origin proposal set.
            pos_priors (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_priors (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following Tensors:

            - labels(Tensor): Gt_labels for all proposals, has
              shape (num_proposals,).
            - label_weights(Tensor): Labels_weights for all proposals, has
              shape (num_proposals,).
            - bbox_targets(Tensor):Regression target for all proposals, has
              shape (num_proposals, 4), the last dimension 4
              represents [tl_x, tl_y, br_x, br_y].
            - bbox_weights(Tensor):Regression weights for all proposals,
              has shape (num_proposals, 4).
        """
        num_pos = pos_priors.size(0)
        num_neg = neg_priors.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        # labels = pos_priors.new_full((num_samples, ),
        #                              self.num_classes,
        #                              dtype=torch.long)
        labels = pos_priors.new_full((num_samples,),
                                     class_number,
                                     dtype=torch.long)
        label_weights = pos_priors.new_zeros(num_samples)
        bbox_targets = pos_priors.new_zeros(num_samples, 4)
        bbox_weights = pos_priors.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_priors, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1
        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results: List[SamplingResult],
                    rcnn_train_cfg: ConfigDict,
                    class_number,
                    concat: bool = True) -> tuple:
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_targets_single` function.

        Args:
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

            - labels (list[Tensor],Tensor): Gt_labels for all
              proposals in a batch, each tensor in list has
              shape (num_proposals,) when `concat=False`, otherwise just
              a single tensor has shape (num_all_proposals,).
            - label_weights (list[Tensor]): Labels_weights for
              all proposals in a batch, each tensor in list has shape
              (num_proposals,) when `concat=False`, otherwise just a
              single tensor has shape (num_all_proposals,).
            - bbox_targets (list[Tensor],Tensor): Regression target
              for all proposals in a batch, each tensor in list has
              shape (num_proposals, 4) when `concat=False`, otherwise
              just a single tensor has shape (num_all_proposals, 4),
              the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            - bbox_weights (list[tensor],Tensor): Regression weights for
              all proposals in a batch, each tensor in list has shape
              (num_proposals, 4) when `concat=False`, otherwise just a
              single tensor has shape (num_all_proposals, 4).
        """
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_priors_list = [res.pos_priors for res in sampling_results]
        neg_priors_list = [res.neg_priors for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_targets_single,
            pos_inds_list,
            neg_inds_list,
            pos_priors_list,
            neg_priors_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            class_number=class_number,
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    def refine_bboxes(self, sampling_results: Union[List[SamplingResult],
                                                    InstanceList],
                      bbox_results: dict,
                      batch_img_metas: List[dict]) -> InstanceList:
        """Refine bboxes during training.

        Args:
            sampling_results (List[:obj:`SamplingResult`] or
                List[:obj:`InstanceData`]): Sampling results.
                :obj:`SamplingResult` is the real sampling results
                calculate from bbox_head, while :obj:`InstanceData` is
                fake sampling results, e.g., in Sparse R-CNN or QueryInst, etc.
            bbox_results (dict): Usually is a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `rois` (Tensor): RoIs with the shape (n, 5) where the first
                  column indicates batch id of each RoI.
                - `bbox_targets` (tuple):  Ground truth for proposals in a
                  single image. Containing the following list of Tensors:
                  (labels, label_weights, bbox_targets, bbox_weights)
            batch_img_metas (List[dict]): List of image information.

        Returns:
            list[:obj:`InstanceData`]: Refined bboxes of each image.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import numpy as np
            >>> from mmdet.models.task_modules.samplers.
            ... sampling_result import random_boxes
            >>> from mmdet.models.task_modules.samplers import SamplingResult
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            ... batch_img_metas = [{'img_shape': (scale, scale)}
            >>>                     for _ in range(n_img)]
            >>> sampling_results = [SamplingResult.random(rng=10)
            ...                     for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torch.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torch.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torch.randint(0, 81, (scale,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> cls_score = torch.randn((scale, 81))
            ... # For each image, pretend random positive boxes are gts
            >>> bbox_targets = (labels, None, None, None)
            ... bbox_results = dict(rois=rois, bbox_pred=bbox_preds,
            ...                     cls_score=cls_score,
            ...                     bbox_targets=bbox_targets)
            >>> bboxes_list = self.refine_bboxes(sampling_results,
            ...                                  bbox_results,
            ...                                  batch_img_metas)
            >>> print(bboxes_list)
        """
        pos_is_gts = [res.pos_is_gt for res in sampling_results]
        # bbox_targets is a tuple
        labels = bbox_results['bbox_targets'][0]
        cls_scores = bbox_results['cls_score']
        rois = bbox_results['rois']
        bbox_preds = bbox_results['bbox_pred']
        if self.custom_activation:
            # TODO: Create a SeasawBBoxHead to simplified logic in BBoxHead
            cls_scores = self.loss_cls.get_activation(cls_scores)
        if cls_scores.numel() == 0:
            return None
        # if cls_scores.shape[-1] == self.num_classes + 1:
        #     # remove background class
        #     cls_scores = cls_scores[:, :-1]
        # elif cls_scores.shape[-1] != self.num_classes:
        #     raise ValueError('The last dim of `cls_scores` should equal to '
        #                      '`num_classes` or `num_classes + 1`,'
        #                      f'but got {cls_scores.shape[-1]}.')
        labels = torch.where(labels == self.num_classes, cls_scores.argmax(1),
                             labels)

        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(batch_img_metas)

        results_list = []
        for i in range(len(batch_img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = batch_img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep
            results = InstanceData(bboxes=bboxes[keep_inds.type(torch.bool)])
            results_list.append(results)

        return results_list
