#!/usr/bin/env python
# coding: utf-8
from pycocotools.coco import COCO
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import pathlib
import json
import random
random.seed(0)

class GetSplitDataset():
    def __init__(self, input_path):
        annotation = input_path # annotation 경로
        self.coco = COCO(annotation)
        self.annotations = dict(self.coco.anns)

    def get_dataframes(self, test_size=0.1, random_state=0):
        images = dict()
        for i in tqdm(self.annotations, desc="Get annotations"):
            if self.annotations[i]['image_id'] not in images:
                row = dict()
                for cat in range(len(self.coco.cats)):
                    row[cat] = False
                images[self.annotations[i]['image_id']] = row
            images[self.annotations[i]['image_id']][self.annotations[i]['category_id']] = True

        df = pd.DataFrame(columns=['img_id'] + [i for i in range(len(self.coco.cats))])

        for idx in tqdm(images, desc="Get image ids"):
            row = images[idx]
            row['img_id'] = idx
            df = df.append(row, ignore_index=True)

        X_train, X_test, y_train, y_test = train_test_split(df, df[[i for i in range(len(self.coco.cats))]], test_size=test_size, random_state=random_state, shuffle=True, stratify=df[[8,9]])
        X_train, X_test, y_train, y_test = X_train.sort_index(), X_test.sort_index(), y_train.sort_index(), y_test.sort_index()
        return X_train, X_test

    def get_dataframe_ann(self):
        self.df_ann = pd.DataFrame(columns=['img_id', 'category'])
        for i in tqdm(self.annotations):
            self.df_ann = self.df_ann.append({'img_id':self.annotations[i]['image_id'], 'category': self.annotations[i]['category_id']}, ignore_index=True)

    def makeJson(self, df, filename):
        json_config = {
            'info': self.coco.dataset['info'],
            'licenses': self.coco.dataset['licenses'],
            'images':[],
            'categories': self.coco.dataset['categories'],
            'annotations':[]
        }
        for idx in df['img_id'].unique():
            json_config['images'].append(self.coco.dataset['images'][idx])
        for idx in df.index:
            for ann_idx in self.df_ann[self.df_ann['img_id'] == idx].index:
                json_config['annotations'].append(self.coco.dataset['annotations'][ann_idx])
        with open(filename, 'w') as fp:
            json.dump(json_config, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset - File names')
    parser.add_argument(
        '--input', 
        type=pathlib.Path,
        default='./train.json'
    )
    parser.add_argument(
        '--train', 
        type=pathlib.Path,
        default='split_train.json'
    )
    parser.add_argument(
        '--valid', 
        type=pathlib.Path,
        default='split_valid.json'
    )
    args = parser.parse_args()

    # start
    get_dataset = GetSplitDataset(input_path=args.input)
    X_train, X_test = get_dataset.get_dataframes(
        test_size=0.1,
        random_state=0
    )
    get_dataset.get_dataframe_ann()
    get_dataset.makeJson(X_train, filename=args.train)
    get_dataset.makeJson(X_test, filename=args.valid)