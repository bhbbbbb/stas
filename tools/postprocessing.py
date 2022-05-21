import os
from multiprocessing import Process
from typing import List

from tqdm import tqdm
import torch
from torchvision import io
from spot_size_est.bfs import BFS


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
    PATH = 'D:\Documents\PROgram\ML\kaggle\stas-seg\kaggle_output\public_inf'
    fix_noise(PATH, num_workers=4)
