# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union, List

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from torch import Tensor

from mmdet.registry import MODELS
from .bbox_head import BBoxHead
from .convfc_bbox_head import ConvFCBBoxHead
from mmdet.models.task_modules.samplers import SamplingResult

import torch
import numpy as np


@MODELS.register_module()
class Shared2FCBBoxHeadScaleDetSimilarity(ConvFCBBoxHead):

    def __init__(self, fc_out_channels: int = 1024, *args, **kwargs) -> None:
        super().__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        self.cls_feature_mapping = nn.Linear(1024,
                  512, bias=True)


        self.text_feature_mapping = nn.Linear(512,
                                             512, bias=True)

        # self.text_embedding = torch.load("./files/NWPU_embedding_11classes.pt")
        self.text_embedding = torch.load("./files/NWPU_embedding_by_prompt_11classes.pt")
        # self.text_embedding = torch.load("./files/NWPU_embedding_by_prompt_11classes_addnorm.pt")
        # self.text_embedding = torch.load("./files/NWPU_embedding_by_prompt_11classes_no_project.pt")

        self.text_embedding = self.text_embedding.cuda()
        self.text_embedding = self.text_embedding.float()
        # self.text_embedding.requires_grad_(False)
        self.text_embedding = self.text_embedding.detach()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.semantic_similarities = torch.load("./files/NWPU_embedding_similarity_11classes.pt")
        self.semantic_similarities = self.semantic_similarities.cuda()
        self.semantic_similarities = self.semantic_similarities.float()
        self.semantic_similarities = self.semantic_similarities.detach()

        # background = torch.zeros(1, 10).cuda()
        #
        # self.semantic_similarities = torch.cat((self.semantic_similarities, background), dim=0)
        # self.semantic_similarities = self.semantic_similarities.detach()

        self.MSE_loss = nn.MSELoss()
        self.MSE_loss_factor = 1


    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """

        # print(x.size())

        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # print("x_cls.size: ", x_cls.size())
        # exit(0)

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None


        cls_mapping = self.cls_feature_mapping(x_cls)

        feature_after_map = cls_mapping / cls_mapping.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        # logits = logit_scale * feature_after_map @ self.text_embedding.t()

        # 增加文本特征映射---start
        text_embedding_after_map = self.text_feature_mapping(self.text_embedding)
        text_embedding_after_map = text_embedding_after_map /text_embedding_after_map.norm(dim=1,keepdim=True)
        logits = logit_scale * feature_after_map @ text_embedding_after_map.t()

        # 增加文本特征映射---end

        # print(cls_score.size())
        # print(cls_mapping.size())
        #
        # print(logits.size())
        #
        # exit(0)

        cls_score = logits


        return cls_score, bbox_pred


    def loss_and_target(self,
                        cls_score: Tensor,
                        bbox_pred: Tensor,
                        rois: Tensor,
                        sampling_results: List[SamplingResult],
                        rcnn_train_cfg: ConfigDict,
                        concat: bool = True,
                        reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the features extracted by the bbox head.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss and targets components.
                The targets are only used for cascade rcnn.
        """

        cls_reg_targets = self.get_targets(
            sampling_results, rcnn_train_cfg, concat=concat)

        # print("cls_reg_targets: ", cls_reg_targets)
        # print(cls_reg_targets[0].size())
        # labels = cls_reg_targets[0]
        # print(labels[:10])
        # print(labels[512:522])
        # print(sampling_results)
        # exit(0)
        labels = cls_reg_targets[0]

        similarity_label = self.semantic_similarities[labels]



        loss_similarity =  self.MSE_loss_factor * self.MSE_loss(cls_score, similarity_label)

        losses = self.loss(
            cls_score,
            bbox_pred,
            rois,
            *cls_reg_targets,
            reduction_override=reduction_override)

        # print(losses)
        losses['loss_similarity'] = loss_similarity

        # cls_reg_targets is only for cascade rcnn
        return dict(loss_bbox=losses, bbox_targets=cls_reg_targets)



