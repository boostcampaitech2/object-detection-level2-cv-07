#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install --ignore-installed mmcv-full==1.3.14 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html


# In[1]:


# 모듈 import

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)


# In[2]:


classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
cfg = Config.fromfile('./configs/_base_/my_train/cascade_rcnn_swin-t-p4-w7_fpn_40epo_coco.py')

root='../dataset/'

#dataset config 수정
cfg.optimizer.type='Adam'
cfg.optimizer.lr=0.00001
del(cfg.optimizer.momentum)

cfg.data.train.classes = classes
cfg.data.train.img_prefix = root
cfg.data.train.ann_file = root + 'train.json' # train json 정보
cfg.data.train.pipeline[2]['img_scale'] = (1024,1024) # Resize

cfg.data.test.classes = classes
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = root + 'test.json' # test json 정보
cfg.data.test.pipeline[1]['img_scale'] = (1024,1024) # Resize

cfg.data.samples_per_gpu = 4

cfg.seed = 2021
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs/cascade_rcnn_x101_64x4d_swin-t-p4-w7_fpn_40epo_coco.py'

cfg.model.roi_head.bbox_head[0].num_classes = 10
cfg.model.roi_head.bbox_head[1].num_classes = 10
cfg.model.roi_head.bbox_head[2].num_classes = 10

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)


# In[3]:


print(cfg.optimizer)


# In[4]:


# build_dataset
datasets = [build_dataset(cfg.data.train)]


# In[5]:


# dataset 확인
datasets[0]


# In[6]:


# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
model.init_weights()


# In[7]:


# 모델 학습
train_detector(model, datasets[0], cfg, distributed=False, validate=False)

