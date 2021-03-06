# Level2 P-stage Object Detection

### π‘ **Team: μ»΄ν¨ν°κ΅¬μ‘°**

## Project Overview

- **Predict Trash Objects**
- Input: 1024 x 1024 Image
- Output: Object annotations
    - Class(10): General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
    - Confidence score: 0 ~ 1
    - Object coordinate: (xmin, ymin, xmax, ymax)

## Archive contents

```
image-classification-level1-02/
βββ dataset/
β   βββ train/ (4883 images)
β   βββ test/  (4871 images)
β   βββ train.json
β   βββ test.json
βββ mmdetection/
β   βββ configs/
β   βββ mmdetection library files
β   βββ train.py
β   βββ inference.py
βββ util/
```

- ```dataset/``` : download from [https://stages.ai/](https://stages.ai/)

## get start

### train & inference
```
cd mmdetection

python train.py
python inference.py
```

### visualize
```
cd util

jupyter notebook Visualize.ipynb

set result csv in second shell
```
it also has Visualize_val_gt -> Visualize ground truth of train data & valid

it also has Visualize_val -> Visualize result of valid

### Requirements

- Ubuntu 18.04.5
- Python 3.8.5
- pytorch 1.7.1
- torchvision 0.8.2

Install packages :  `pip install -r requirements.txt` 

#### Hardware

- CPU: 8 x Intel(R) Xeon(R) Gold 5220 CPU
- GPU: V100
- RAM: 88GB


## Contributors

| **Name** @github                                              | 
| ------------------------------------------------------------  | 
| **κ³ μ¬μ±** [@κ³ μ¬μ±](https://github.com/pkpete)               |
| **κΉμ±λ―Ό** [@ksm0517](https://github.com/ksm0517)             |
| **λ°μ§λ―Ό** [@λ°μ§λ―Ό](https://github.com/ddeokbboki-good)      | 
| **λ°μ§ν** [@ppjh8263](https://github.com/ppjh8263)           |
| **μ¬μΈλ Ή** [@Seryoung Shim](https://github.com/seryoungshim17)| 
| **μ€νμ ** [@Yoon Hajung](https://github.com/YHaJung)         | 

## Data Citation ```λ€μ΄λ² μ»€λ₯νΈμ¬λ¨ - μ¬νμ© μ°λ κΈ° λ°μ΄ν°μ / CC BY 2.0```
