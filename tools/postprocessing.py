import os
import json
from multiprocessing import Process, Pool
from typing import List, Dict

from tqdm import tqdm
import torch
from torchvision import io
from spot_size_est.bfs import BFS #, ParallelBFS
from spot_size_est.spot_validate.nfnet_model_utils import NfnetModelUtils, NfnetConfig, ROI
from spot_size_est.spot_validate.dataset import DatasetConfig

ROI_EDGE_LEN = 192 # F0
# ROI_EDGE_LEN = 224 # F1

class Config(NfnetConfig, DatasetConfig):
    learning_rate: float = 0
    pin_memory: bool = True
    small_spot_threshold: int = 200
    # nf_variant: str
    nf_batch_size_inf: int = 8
    num_workers: int = 4
    

def load_nfnet_utils():
    WEIGHT_PATH = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\src\\log\\20220522T15-58-52_F0\\20220522T21-54-56_epoch_46'
    config = Config()
    config.valid_spot_exp_val_threshold = 0
    config.check_and_freeze()
    config.display()
    model = NfnetModelUtils.init_model(config)
    utils = NfnetModelUtils.load_checkpoint(model, WEIGHT_PATH, config)
    return utils


def fix_noise_and_valid_spot(
    mask_dir: str,
    img_dir: str,
    out_dir: str = None,
    num_workers: int = 4,
    debugging: bool = False,
):
    if out_dir is None:
        out_dir = os.path.join(mask_dir, '..', os.path.basename(mask_dir) + '_fixed_v')
    os.makedirs(out_dir, exist_ok=True)
    names = [name[:-4] for name in os.listdir(mask_dir) if name.endswith('.png')]
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
                for roi, to_rm_spot, exp_val in zip(rois, to_rm_spots, exp_vals):
                    roi: ROI
                    debugging_output[name].append({
                        'roi': roi._asdict(),
                        'exp_val': exp_val,
                        'confidence': exp_val / roi.size,
                        'is_rm': to_rm_spot,
                    })

            mask = fixer.fill_black(
                [roi_mask for roi_mask, to_rm in zip(roi_masks, to_rm_spots) if to_rm]
            )
            mask = torch.from_numpy(mask).unsqueeze(0)
            io.write_png(mask, os.path.join(out_dir, name + '.png'))
    if debugging:
        with open('roi_debug.json', 'w', encoding='utf-8') as fout:
            json.dump(debugging_output, fout, indent=4)
    return

def _do_bfs(idx_mask):
    idx, mask = idx_mask
    fixer = BFS(mask)
    rois, roi_masks = fixer.get_rois(ROI_EDGE_LEN, fill_white=True, require_roi_spot_mask=True)
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
        mask = fixer.fill_noise(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        io.write_png(mask, os.path.join(out_dir, mask_name))
    return

if __name__ == '__main__':
    MASK_DIR = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\kaggle_output\\valid_inf'
    IMG_DIR = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\SEG_Train_Datasets\\Train_Images'
    # PATH = 'D:\Documents\PROgram\ML\kaggle\stas-seg\kaggle_output\public_inf'
    # no fix # 0.8647
    # fix_noise(MASK_DIR, num_workers=4) # 0.8676
    # 0.839664 1000 .25 .75
    # 0.854180 0 .25 .75
    fix_noise_and_valid_spot(MASK_DIR, IMG_DIR, num_workers=6, debugging=True)
    
