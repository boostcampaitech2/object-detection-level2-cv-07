from pycocotools.coco import COCO
import pandas as pd
import json
import argparse
import pathlib
import os

def csvToCoco(target_file, csv_file):
    coco = COCO(target_file)
    predictions = {
        "images": coco.dataset["images"].copy(),
        "categories": coco.dataset["categories"].copy(),
        "annotations": None
    }
    
    df = pd.read_csv(csv_file)
    annotations = []
    for row_idx, row in df.iterrows():
        try:
            for idx, pred in enumerate(row['PredictionString'].split()):
                if idx % 6 == 0:
                    bbox = []
                    ann = dict(
                        image_id=int(row['image_id']),
                        iscrowd=0,
                        id=row_idx + idx,
                        category_id = int(pred)
                    )
                elif idx % 6 == 1:
                    ann['score'] = float(pred)
                else:
                    bbox.append(float(pred))

                if idx % 6 == 5:
                    ann['bbox'] = bbox
                    ann['area'] = bbox[-1] * bbox[-2]
                    # print(ann)
                    annotations.append(ann)
        except:
            pass
    predictions['annotations'] = annotations
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'predict_valid.json'), 'w') as f:
        json.dump(predictions, f)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='csv to coco')
    parser.add_argument(
        '--csv_path', 
        type=pathlib.Path,
        default='./retina_base_epoch_19_thr1.csv'
    )
    args = parser.parse_args()
    
    csvToCoco(os.path.join('/opt/ml/detection/dataset/','split_valid.json'), args.csv_path)
    print(args.csv_path)