#!/usr/bin/env python
# coding: utf-8

# In[7]:


# 모듈 import

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)


# In[8]:


classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
cfg = Config.fromfile('./configs/cbnet/cascade_mask_rcnn_cbv2_swin_3x_coco.py')

root='../dataset/'

# dataset config 수정
cfg.data.train.classes = classes
cfg.data.train.img_prefix = root
cfg.data.train.ann_file = root + 'split_train.json' # train json 정보
cfg.data.train.pipeline[2]['img_scale'] = (1024,1024) # Resize

cfg.data.val.classes = classes
cfg.data.val.img_prefix = root
cfg.data.val.ann_file = root + 'split_valid.json' # test json 정보
cfg.data.test.pipeline[1]['img_scale'] = (1024,1024) # Resize

cfg.data.samples_per_gpu = 4
cfg.data.workers_per_gpu = 1

cfg.seed = 2021
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs/CBSwinTransformer'

cfg.model.roi_head.bbox_head.num_classes = 10

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
cfg.total_epochs = 50

cfg.workflow = [('train', 1), ('val', 1)]
cfg.lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
# In[9]:


# build_dataset
datasets = [
    build_dataset(cfg.data.train), 
    build_dataset(cfg.data.val)
]


# In[10]:


# dataset 확인
datasets[0]


# In[11]:


# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
model.init_weights()


# In[ ]:


# 모델 학습
train_detector(model, datasets[0], cfg, distributed=False, validate=True)

