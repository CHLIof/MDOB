
load_from = None
lang_model_name = 'bert-base-uncased'


# dataset settings

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None



model = dict(
    type='MDOB',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=r"/home/ly/workspace/unifymodel/mmdet/model/swin_tiny_patch4_window7_224.pth")
    ),
    neck=dict(
        type='FPN_DropBlock',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        relu_before_extra_convs=True,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSVLFusionHeadProject2ParamInit_KD',
        lang_model_name=lang_model_name,
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        dataset_quantity=3,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoderForGLIP',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    language_model=dict(type='BertModel', name=lang_model_name),
    teacher_model=dict(path='/home/newspace/ly/workspace/mmdet/output/TeacherModel.pth'),
    train_cfg=dict(
        assigner=dict(
            type='ATSSAssigner',
            topk=9,
            iou_calculator=dict(type='BboxOverlaps2D_GLIP')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        imdecode_backend='pillow',
        backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='GTBoxSubOne_GLIP'),
    dict(
        type='FixScaleResize',
        scale=(1333, 800),
        keep_ratio=True,
        backend='pillow'),
    dict(type='RandomFlip_GLIP', prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'dataset_num'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=backend_args,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(1333, 800),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities', 'dataset_num'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler2', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='ConcatDataset',
            ignore_keys=['dataset_type'],
            datasets=[

                dict(
                    type='RSODDataset',
                    data_root=r"/home/ly/workspace/dataset/",
                    ann_file='RSOD_coco/train.json',
                    data_prefix=dict(img='RSOD_coco/images/'),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32),
                    pipeline=train_pipeline,
                    return_classes=True,
                    dataset_num = 0,
                    backend_args=backend_args),
                dict(
                    type='UCASDataset',
                    data_root=r"/home/ly/workspace/dataset/",
                    ann_file='UCAS/train.json',
                    data_prefix=dict(img='UCAS/images/'),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32),
                    pipeline=train_pipeline,
                    return_classes=True,
                    dataset_num=1,
                    backend_args=backend_args),
                dict(
                    type='NWPUDataset',
                    data_root=r"/home/ly/workspace/dataset/",
                    ann_file='NWPU_VHR-10_dataset/train.json',
                    data_prefix=dict(img='NWPU_VHR-10_dataset/positive_image set/'),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32),
                    pipeline=train_pipeline,
                    return_classes=True,
                    dataset_num = 2,
                    backend_args=backend_args),
            ]
        )
    ))

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
                 # return_classes=True,
                 type='ConcatDataset',
                 datasets = [
                     dict(
                         type='RSODDataset',
                         data_root=r"/home/ly/workspace/dataset/",
                         ann_file='RSOD_coco/val.json',
                         data_prefix=dict(img='RSOD_coco/images/'),
                         return_classes=True,
                         pipeline=test_pipeline,
                         dataset_num = 0,
                         test_mode=True,
                         backend_args=backend_args),
                     dict(
                         type='UCASDataset',
                         data_root=r"/home/ly/workspace/dataset/",
                         ann_file='UCAS/val.json',
                         data_prefix=dict(img='UCAS/images/'),
                         return_classes=True,
                         pipeline=test_pipeline,
                         dataset_num=1,
                         test_mode=True,
                         backend_args=backend_args),
                     dict(
                         type='NWPUDataset',
                         data_root=r"/home/ly/workspace/dataset/",
                         ann_file='NWPU_VHR-10_dataset/val.json',
                         data_prefix=dict(img='NWPU_VHR-10_dataset/positive_image set/'),
                         return_classes=True,
                         pipeline=test_pipeline,
                         dataset_num=2,
                         test_mode=True,
                         backend_args=backend_args),
                 ]
                 ))
test_dataloader = val_dataloader


val_evaluator_NWPU = dict(
    type='CocoMetric',
    ann_file="/home/ly/workspace/dataset/NWPU_VHR-10_dataset/val.json",
    metric='bbox')
val_evaluator_RSOD = dict(
    type='CocoMetric',
    ann_file="/home/ly/workspace/dataset/RSOD_coco/val.json",
    metric='bbox')
val_evaluator_UCAS = dict(
    type='CocoMetric',
    ann_file="/home/ly/workspace/dataset/UCAS/val.json",
    metric='bbox')
metrics = [val_evaluator_NWPU, val_evaluator_RSOD, val_evaluator_UCAS]
dataset_prefixes = ['NWPU', 'RSOD', 'UCAS']

val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    # ann_file=data_root + 'val.json',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes
    # format_only=False,
    # backend_args=backend_args
)
test_evaluator = val_evaluator

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[24, 33],
        gamma=0.1)
]

evaluation = dict(
    metric='bbox',
    save_best = 'bbox_mAP',
)

# We did not adopt the official 24e optimizer strategy
# because the results indicate that the current strategy is superior.
optim_wrapper = dict(

    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    clip_grad=None)


auto_scale_lr = dict(enable=False, base_batch_size=16)

default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_last=True, save_best='UCAS/coco/bbox_mAP', rule="greater"),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', score_thr = 0.5))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'

resume = False
work_dir="/home/newspace/ly/workspace/mmdet/output/exp_test43"








