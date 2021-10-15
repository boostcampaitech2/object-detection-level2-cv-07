#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np

from torch.utils.data import DataLoader, Dataset
import cv2
import torch


# In[2]:


# config file 들고오기
# cfg = Config.fromfile('./configs/_base_/retina_base/retinanet_r50_fpn_2x_coco.py')
cfg = Config.fromfile('/opt/ml/object-detection-level2-cv-07/mmdetection/configs/_base_/retina_base/retinanet_r50_fpn_2x_coco.py')

root='/opt/ml/detection/dataset/'

epoch = 'epoch_19'

# dataset config 수정
# cfg.data.test.classes = classes
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = root + 'train.json'
cfg.data.test.pipeline[1]['img_scale'] = (1024,1024) # Resize
cfg.data.test.test_mode = True

# cfg.data.samples_per_gpu = 4

# dataset config 수정
cfg.model.bbox_head.num_classes = 10
cfg.seed = 2021
cfg.gpu_ids = [0]
cfg.work_dir = '/opt/ml/object-detection-level2-cv-07/mmdetection/work_dirs/base_retinanet_fpn_2x_coco'

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.model.train_cfg = None


# In[3]:


# build dataset & dataloader
dataset = build_dataset(cfg.data.val)
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)


# In[4]:


# checkpoint path
checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

model.CLASSES = dataset.CLASSES
model = MMDataParallel(model.cuda(), device_ids=[0])


# In[5]:


# output = single_gpu_test(model, data_loader, show_score_thr=0.05) # output 계산
output = single_gpu_test(model, data_loader, show_score_thr=0.1)


# In[6]:


# submission 양식에 맞게 output 후처리
prediction_strings = []
file_names = []
coco = COCO(cfg.data.test.ann_file)
img_ids = COCO(cfg.data.val.ann_file).getImgIds()

class_num = 10
for idx, out in zip(img_ids, output):
    prediction_string = ''
    image_info = coco.loadImgs(coco.getImgIds(imgIds=idx))[0]
    for j in range(class_num):
        for o in out[j]:
            prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                o[2]) + ' ' + str(o[3]) + ' '
        
    prediction_strings.append(prediction_string)
    file_names.append(image_info['id'])


submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv(os.path.join(cfg.work_dir, f'retina_base_{epoch}_thr1.csv'), index=None)
submission.head()


# In[ ]:




