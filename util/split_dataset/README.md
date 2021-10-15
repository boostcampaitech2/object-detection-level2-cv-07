## Overview
- Split `train.json` into `split_train.json` / `split_valid.json`

## How to use
### 1. Install COCO API
- Windows
```bash
pip install Cython
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```
- Linux
```bash
pip install pycocotools
```
### 2. Execute
```bash
sh run.sh
```
- Change ```train.json file path```
#### **`run.sh`**
```bash
--input '{train.json file path}' \
```
- Change ```output json file names```
#### **`run.sh`**
```bash
--train '{train dataset json file}' \
--valid '{validation dataset json file}'
```