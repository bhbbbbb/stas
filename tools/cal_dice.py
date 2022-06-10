import os

from tqdm import tqdm

import torch
from torch import Tensor
from torchvision import io


MASK_DIR = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\SEG_Train_Datasets\\Train_Masks'


def dice(preds: Tensor, labels: Tensor) -> Tensor:
    tp = torch.sum(labels*preds)
    fn = torch.sum(labels*(1-preds))
    fp = torch.sum((1-labels)*preds)
    score = (tp / (tp + 0.5 * (fn + fp))).item()
    return tp.item(), fn.item(), fp.item(), score
    
def cal_dice_score(test_dir: str):
    mask_names = [name for name in os.listdir(test_dir) if name.endswith('.png')]
    mask_paths = [os.path.join(test_dir, name) for name in mask_names]
    gt_mask_paths = [os.path.join(MASK_DIR, name) for name in mask_names]

    # score = 0.0
    tp = 0.0
    fp = 0.0
    fn = 0.0
    step = 0
    for mask_path, gt_mask_path in tqdm(zip(mask_paths, gt_mask_paths), total=len(mask_names)):
        step += 1
        mask = io.read_image(mask_path)
        mask.div_(255, rounding_mode='trunc')
        gt_mask = io.read_image(gt_mask_path)
        tp_, fn_, fp_, _ = dice(mask, gt_mask)
        # score += score_
        tp += tp_
        fp += fp_
        fn += fn_
    # score /= step
    score = tp / (tp + 0.5 * (fn + fp))
    print('score: ', score)
    print(f'tp: {tp}, fp: {fp}, fn: {fn}')

    return

# if __name__ == '__main__':
#     TEST_DIR = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\kaggle_output\\valid_inf_fixed'
#     cal_dice_score(TEST_DIR)
    