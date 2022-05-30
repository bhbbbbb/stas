import os
import json
from multiprocessing import Process, Pool
from typing import List, Dict

from tqdm import tqdm
import pandas as pd
import torch
from torch import Tensor
from torchvision import io
from spot_validate.bfs import BFS
from spot_validate.nfnet_model_utils import NfnetModelUtils, NfnetConfig, ROI
# from spot_validate.roinfnet_model_utils import RoiNfnetModelUtils
from spot_validate.dataset import DatasetConfig

ROI_EDGE_LEN = 192 # F0
# ROI_EDGE_LEN = 224 # F1
MIN_SPOT = 450
MAX_ROI_SIZE = 2000
DATASET_ROOT: str = os.path.join(__file__, '..', '..', '..', 'SEG_Train_Datasets')
GT_MASK_ROOT: str = os.path.join(DATASET_ROOT, 'Train_Masks')

class Config(NfnetConfig, DatasetConfig):
    learning_rate: float = 0
    pin_memory: bool = True
    small_spot_threshold: int = 200
    # nf_variant: str
    nf_batch_size_inf: int = 1
    batch_size: dict = {
        'train': 1,
        'val': 1,
    }
    num_workers: int = 4
    valid_spot_confidence_lower_bound: float = 0.0
    # valid_spot_confidence_upper_bound: float = 0.036
    # valid_spot_exp_val_threshold: float = 45.0
    valid_spot_confidence_upper_bound: float = 0.1
    valid_spot_exp_val_threshold: float = 500.0
    train_img_len: int = ROI_EDGE_LEN

def load_nfnet_utils():
    # WEIGHT_PATH = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\src\\log\\20220522T15-58-52_F0\\20220526T18-45-25_epoch_137'
    # WEIGHT_PATH = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\src\\log\\20220522T15-58-52_F0\\20220526T02-48-32_epoch_95'
    # WEIGHT_PATH = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\src\\log\\20220522T15-58-52_F0\\20220530T01-51-05_epoch_149'
    WEIGHT_PATH = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\src\\log\\20220522T15-58-52_F0\\20220530T20-55-55_epoch_164'
    config = Config()
    config.nf_variant = 'F0'
    # config.max_roi_len = 80
    # config.train_img_len = 256
    config.train_img_len = 192
    config.check_and_freeze()
    config.display()
    model = NfnetModelUtils.init_model(config)
    utils = NfnetModelUtils.load_checkpoint(model, WEIGHT_PATH, config)
    # model = RoiNfnetModelUtils.init_model(config)
    # utils = RoiNfnetModelUtils.load_checkpoint(model, WEIGHT_PATH, config)
    return utils


def fix_noise_and_valid_spot(
    mask_dir: str,
    img_dir: str,
    out_dir: str = None,
    num_workers: int = 4,
    num_out: int = None,
    debugging: bool = False,
):
    if out_dir is None:
        out_dir = os.path.join(mask_dir, '..', os.path.basename(mask_dir) + '_fixed_v')
    os.makedirs(out_dir, exist_ok=True)
    names = [name[:-4] for name in os.listdir(mask_dir) if name.endswith('.png')]
    if num_out is not None:
        names = names[:num_out]
    utils = load_nfnet_utils()
    mask_paths = [os.path.join(mask_dir, name + '.png') for name in names]
    img_paths = [os.path.join(img_dir, name + '.jpg') for name in names]

    get_mask = ((idx, io.read_image(path)) for idx, path in enumerate(mask_paths))
    debugging_output: Dict[str, List[dict]] = {}

    with Pool(processes=num_workers) as pool:
        pbar = tqdm(
            pool.imap_unordered(_do_bfs, get_mask),
            total=len(mask_paths),
        )
        for idx, rois, roi_masks, fixer in pbar:
            fixer: BFS
            name = names[idx]
            rois = [ROI(**roi) for roi in rois]
            to_rm_spots, exp_vals = utils.validate_roi(img_paths[idx], rois, debugging)
            if debugging:
                debugging_output[name] = []
                ground_truth_mask_path = os.path.join(GT_MASK_ROOT, name + '.png')
                gt_mask = io.read_image(ground_truth_mask_path)
                for roi, roi_mask, to_rm_spot, exp_val in\
                                                zip(rois, roi_masks, to_rm_spots, exp_vals):
                    roi: ROI
                    roi_mask: Tensor
                    debugging_output[name].append({
                        'roi': roi._asdict(),
                        'exp_val': exp_val,
                        'confidence': exp_val / roi.size,
                        'is_rm': to_rm_spot,
                        'ground_truth': (gt_mask * roi_mask).sum().item(),
                    })

            mask = fixer.fill_black(
                [roi_mask for roi_mask, to_rm in zip(roi_masks, to_rm_spots) if to_rm]
            )
            mask = torch.from_numpy(mask).unsqueeze(0)
            io.write_png(mask, os.path.join(out_dir, name + '.png'))
    if debugging:
        with open('roi_debug.json', 'w', encoding='utf-8') as fout:
            json.dump(debugging_output, fout, indent=4, sort_keys=True)
        _write_roi_debug_csv(debugging_output)
            
    return

def _write_roi_debug_csv(debugging_output = None):
    if debugging_output is None:
        with open('roi_debug.json', 'r', encoding='utf-8') as fin:
            debugging_output = json.load(fin)
    def gen(debugging_output: Dict[str, List[dict]]):
        for name, l in sorted(debugging_output.items()):
            for d in l:
                d['name'] = name
                d['size'] = d['roi']['size']
                yield d

    df = pd.DataFrame(gen(debugging_output))
    df.to_csv('roi_debug.csv')
    return

def _do_bfs(idx_mask):
    idx, mask = idx_mask
    fixer = BFS(mask)
    rois, roi_masks = fixer.get_rois(
        ROI_EDGE_LEN, fill_white=True, require_roi_spot_mask=True, max_size=MAX_ROI_SIZE,
    )
    return idx, rois, roi_masks, fixer

def fix_noise(in_dir: str, out_dir: str = None, num_workers: int = 4):
    if out_dir is None:
        out_dir = os.path.join(in_dir, '..', os.path.basename(in_dir) + '_fixed')
    os.makedirs(out_dir, exist_ok=True)
    mask_names = [name for name in os.listdir(in_dir) if name.endswith('.png')]
    splits = []
    p_list: List[Process] = []
    for i in range(num_workers):
        names = [name for idx, name in enumerate(mask_names) if idx % num_workers == i]
        splits.append(names)
        p_list.append(Process(target=_fix_noise, args=(names, in_dir, out_dir, not bool(i))))
    
    for p in p_list:
        p.start()

    for p in p_list:
        p.join()
    return

def _fix_noise(mask_names: List[str], in_dir: str, out_dir: str, progress_bar: bool):
    mask_paths = [os.path.join(in_dir, name) for name in mask_names]
    if progress_bar:
        pbar = tqdm(zip(mask_names, mask_paths), total=len(mask_paths))
    else:
        pbar = zip(mask_names, mask_paths)
    for mask_name, mask_path in pbar:
        mask = io.read_image(mask_path)
        fixer = BFS(mask)
        # mask = fixer._fill_noise(C.BLACK, fixer.black_fill_threshold)
        mask = fixer.fill_noise(MIN_SPOT)
        mask = torch.from_numpy(mask).unsqueeze(0)
        io.write_png(mask, os.path.join(out_dir, mask_name))
    return

if __name__ == '__main__':
    MASK_DIR = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\kaggle_output\\valid_inf'
    # MASK_DIR = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\kaggle_output\\valid_inf_20'
    IMG_DIR = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\SEG_Train_Datasets\\Train_Images'
    # PATH = 'D:\Documents\PROgram\ML\kaggle\stas-seg\kaggle_output\public_inf'
    # no fix # 0.853473
    # 0.1, 0.1 # 0.854387
    fix_noise(MASK_DIR, num_workers=6) # 0.85417
    # fix_noise_and_valid_spot(MASK_DIR, IMG_DIR, num_workers=6, debugging=True, num_out=None)
    # _write_roi_debug_csv()
    
