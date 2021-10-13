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