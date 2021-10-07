#!/usr/bin/env python
# coding: utf-8


# 모듈 import
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import build_dataloader
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets import build_dataset


# config file 들고오기
# cfg = Config.fromfile('./DetectoRS.py')
cfg = Config.fromfile('./DetectoRS_resnet.py')
# cfg = Config.fromfile('./swin_DetectoRS.py')

# cfg.optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0005)

# img_size = None

# if img_size:
#     cfg.data.train.pipeline[2]['img_scale'] = (img_size,img_size) # Resize
#     cfg.data.val.pipeline[2]['img_scale'] = (img_size,img_size) # Resize


cfg.seed=2021
cfg.gpu_ids = [0]


# cfg.work_dir = './work_dirs/swin_decetors'
cfg.work_dir = './work_dirs/DetectoRS_ResNet_epoch30_grid_clip'

for bbox_head__ in cfg.model.roi_head.bbox_head:
    bbox_head__.num_classes = 10

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)

# build_dataset
datasets = [build_dataset(cfg.data.train),build_dataset(cfg.data.val_loss)]

# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
model.init_weights()

# print(len(datasets))
train_detector(model, datasets, cfg, distributed=False, validate=True)

