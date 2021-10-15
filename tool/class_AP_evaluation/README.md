## Folder structure
```
object-detection-level2-cv-07/tool/class_AP_evaluation
├── data/
│	├── csv_to_coco.py
│	├── MM_valid_to_csv.py
│	├── TV_valid_to_csv.py
│   └── split_valid.json (validation dataset annotation json file)
├── BoundingBox.py
├── BoundingBoxes.py
├── class_AP.py
├── main.py
├── run.sh
└── utils.py
```
## How to use
- Excute in current directory (```class_AP_evaluation```)
```bash
sh run.sh
```
- Can change csv file path ```run.sh```
```python
cd data && python csv_to_coco.py --csv_path '<file path>'
```
- If want to save log
```bash
sh run.sh >> '<txt file name>'
```
- Excute small object evaluation
```bash
sh run-small-obj.sh
```
- Change small object size (area range) ```main-small-object.py```
```python
small_size = (0, 10000)
```
### Result
```
retina_resnet101_epoch_17.csv
                      class AP   TP     FP   Precision  Recall   F1  
----------------------------------------------------------------------
  General trash(399)  0.1659    112    232      0.3256  0.2807  0.3015
          Paper(676)  0.3964    331    347      0.4882  0.4896  0.4889
      Paper pack(93)  0.3597     38     35      0.5205  0.4086  0.4578
           Metal(74)  0.3630     30     41      0.4225  0.4054  0.4138
           Glass(62)  0.3668     28     25      0.5283  0.4516  0.4870
        Plastic(328)  0.2193    111    161      0.4081  0.3384  0.3700
      Styrofoam(105)  0.4084     47     33      0.5875  0.4476  0.5081
    Plastic bag(553)  0.6075    360    230      0.6102  0.6510  0.6299
         Battery(15)  0.0000      0      2      0.0000  0.0000  0.0000
        Clothing(39)  0.3295     15     17      0.4688  0.3846  0.4225
======================================================================
             Average                            0.4360  0.3858  0.4093
```
## [Citation](https://github.com/rafaelpadilla/Object-Detection-Metrics)
```
@Article{electronics10030279,
AUTHOR = {Padilla, Rafael and Passos, Wesley L. and Dias, Thadeu L. B. and Netto, Sergio L. and da Silva, Eduardo A. B.},
TITLE = {A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit},
JOURNAL = {Electronics},
VOLUME = {10},
YEAR = {2021},
NUMBER = {3},
ARTICLE-NUMBER = {279},
URL = {https://www.mdpi.com/2079-9292/10/3/279},
ISSN = {2079-9292},
DOI = {10.3390/electronics10030279}
}
```