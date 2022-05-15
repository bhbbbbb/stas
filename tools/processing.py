# pylint: disable=all
import os
import json

import numpy as np
import cv2
# from PIL import Image
# from torchvision.transforms import functional as TF
from sklearn.model_selection import train_test_split

DATASET_ROOT = '../SEG_Train_Datasets'
TRAIN_PATH = os.path.join(DATASET_ROOT, 'Train_Images')
ANNO_PATH = os.path.join(DATASET_ROOT, 'Train_Annotations')
MASK_PATH = os.path.join(DATASET_ROOT, 'ann_dir')
TRAIN_SPLIT = os.path.join(DATASET_ROOT, 'split_train.json')
VALID_SPLIT = os.path.join(DATASET_ROOT, 'split_valid.json')
SEED = 0xAAAAAAA
os.makedirs(MASK_PATH, exist_ok=True)

def make_mask():
    for jsonfile in os.listdir(ANNO_PATH):
        f = open(os.path.join(ANNO_PATH, jsonfile))
        data = json.load(f)
        mask = np.zeros((data['imageHeight'], data['imageWidth'], 1), dtype=np.uint8)
        for polygan in data['shapes']:
            pts = np.array(polygan['points'], dtype=np.int32)
            cv2.fillPoly(mask, [pts], color=1)
        save_mask_path = jsonfile.split('.')[0] + '.png'
        save_mask_path = os.path.join(MASK_PATH, save_mask_path)
        cv2.imwrite(save_mask_path, mask)

def make_split(train_ratio: float):
    file_set = os.listdir(ANNO_PATH)
    file_set = [(os.path.splitext(file)[0]) for file in file_set]
    train_set, valid_set = train_test_split(file_set, train_size=train_ratio, random_state=SEED)
    def output_splited_set(splited_set: list, name: str):
        file = os.path.join(DATASET_ROOT, name + '.json')
        with open(file, 'w', encoding='utf-8') as fout:
            json.dump(splited_set, fout, indent=4)
        # with open(file, 'w', encoding='utf-8') as fout:
        #     for a_name in set:
        #         fout.write(a_name)
        #         fout.write('\n')

    output_splited_set(train_set, 'split_train')
    output_splited_set(valid_set, 'split_valid')
    return
    
def load_train_split():
    with open(TRAIN_SPLIT, 'r', encoding='utf-8') as fin:
        train_set = json.load(fin)
    train_set = [os.path.join(TRAIN_PATH, img_name) for img_name in train_set]
    return train_set


if __name__ == "__main__":
    make_split(0.8)
