import os
import json

import numpy as np
import cv2
# from PIL import Image
# from torchvision.transforms import functional as TF
from sklearn.model_selection import train_test_split

SEED = 0xAAAAAAA

def make_mask(anno_dir: str, splashed_mask: bool = False, output_dir: str = None):
    """make masks from given annotations.json

    Args:
        splashed_mask (bool, optional): Whether generate splashed masks.
            splashed mask is only for visualization. For training,
            0 represent background, and 1 represent target. Defaults to False.
        output_dir (str, optional): dirname to output. Defaults to MASK_DIR.
    """
    if output_dir is None:
        name = 'Train_Masks' if not splashed_mask else 'Splashed_Masks'
        output_dir = os.path.join(os.path.dirname(anno_dir), name)
    os.makedirs(output_dir, exist_ok=True)
    for jsonfile in os.listdir(anno_dir):
        with open(os.path.join(anno_dir, jsonfile), encoding='ansi') as fin:
            data = json.load(fin)
            mask = np.zeros((data['imageHeight'], data['imageWidth'], 1), dtype=np.uint8)
            for polygan in data['shapes']:
                pts = np.array(polygan['points'], dtype=np.int32)
                cv2.fillPoly(mask, [pts], color=(255 if splashed_mask else 1))
            save_mask_path = jsonfile.split('.')[0] + '.png'
            save_mask_path = os.path.join(output_dir, save_mask_path)
            cv2.imwrite(save_mask_path, mask)

def make_split(anno_dir: str, train_ratio: float, output_dir: str):
    file_set = os.listdir(anno_dir)
    file_set = [(os.path.splitext(file)[0]) for file in file_set]
    train_set, valid_set = train_test_split(file_set, train_size=train_ratio, random_state=SEED)
    def output_splited_set(splited_set: list, name: str):
        file = os.path.join(output_dir, name + '.json')
        with open(file, 'w', encoding='utf-8') as fout:
            json.dump(splited_set, fout, indent=4)

    output_splited_set(train_set, 'split_train')
    output_splited_set(valid_set, 'split_valid')
    return
    
# def load_train_split():
#     with open(TRAIN_SPLIT, 'r', encoding='utf-8') as fin:
#         train_set = json.load(fin)
#     train_set = [os.path.join(TRAIN_DIR, img_name) for img_name in train_set]
#     return train_set

if __name__ == '__main__':
    DATASET_ROOT = '../SEG_Train_Datasets'
    TRAIN_DIR = os.path.join(DATASET_ROOT, 'Train_Images')
    ANNO_DIR = os.path.join(DATASET_ROOT, 'Train_Annotations')
    MASK_DIR = os.path.join(DATASET_ROOT, 'Train_Masks')
    SPLASHED_DIR = os.path.join(DATASET_ROOT, 'Splashed_Masks')
    TRAIN_SPLIT = os.path.join(DATASET_ROOT, 'split_train.json')
    VALID_SPLIT = os.path.join(DATASET_ROOT, 'split_valid.json')

    make_mask(ANNO_DIR, splashed_mask=False, output_dir=MASK_DIR)
    make_mask(ANNO_DIR, splashed_mask=True, output_dir=SPLASHED_DIR)
    # make_split(0.8)
    print('succeeded')
