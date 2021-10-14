########################################################################
########################   backbone setting   ##########################
########################################################################
_base_ = [
    './backbone/cascade_rcnn_r50_fpn.py',
]

########################################################################
#########################   Model setting   ############################
########################################################################
model = dict(
    backbone=dict(
        type='DetectoRS_ResNeXt',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),###
        stage_with_sac=(False, True, True, True),
        output_img=True),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNeXt',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,###
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),###
            stage_with_sac=(False, True, True, True),
            # pretrained='torchvision://ResNeXt-101-32x8d',
             pretrained='open-mmlab://resnext101_32x4d',
            style='pytorch')))


########################################################################
########################   DataSet setting   ###########################
########################################################################
# dataset settings
dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
data_root = './dataset/'
classes_list =  ("General trash", "Paper", "Paper pack",
                    "Metal", "Glass", "Plastic", "Styrofoam",
                    "Plastic bag", "Battery", "Clothing")
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg = dict(
    mean=[123.650, 117.397, 110.075], std=[54.034, 53.369, 54.783], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='Resize', img_scale=[(1024, 1024), (2048, 2048)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'split_train.json',
        classes = classes_list,
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'split_valid.json',
        classes = classes_list,
        img_prefix=data_root,
        pipeline=test_pipeline),
    val_loss=dict(
        type=dataset_type,
        ann_file=data_root + 'split_valid.json',
        classes = classes_list,
        img_prefix=data_root,
        pipeline=valid_pipeline),
    test=dict(
        test_mode = True,
        type=dataset_type,
        ann_file=data_root + 'test.json',
        classes = classes_list,
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')


########################################################################
########################   Scedules setting   ##########################
########################################################################

# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
    custom_keys={
    'absolute_pos_embed': dict(decay_mult=0.),
    'relative_position_bias_table': dict(decay_mult=0.),
    'norm': dict(decay_mult=0.)
}))
# optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0005)

optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
lr_config = dict(
    policy='cyclic',
    target_ratio=(1e-3, 1e-6),
    cyclic_times=3,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)



runner = dict(type='EpochBasedRunner', max_epochs=30)


########################################################################
########################   Runtime setting   ###########################
########################################################################

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook',
        init_kwargs=dict(
            project='DetectoRS',
            name='DetectoRS_cyc_aug'))
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
