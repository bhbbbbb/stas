import os

from tqdm import tqdm
import torch
from torchvision import io
from spot_size_est.bfs import BFS, C

    
def do():
    path = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\src\\log\\20220516T00-37-26\\splash_170_'
    mask_names = os.listdir(path)
    masks_path = [os.path.join(path, m) for m in mask_names]
    out_dir = os.path.join(path, '..', os.path.basename(path) + '_fixed')
    os.makedirs(out_dir, exist_ok=True)
    for mask_name, mask_path in tqdm(zip(mask_names, masks_path), total=len(masks_path)):
        mask = io.read_image(mask_path)
        fixer = BFS(mask)
        # mask = fixer._fill_noise(C.BLACK, fixer.black_fill_threshold)
        mask = fixer.fill_noise(0, True)
        mask = torch.from_numpy(mask).unsqueeze(0)
        io.write_png(mask, os.path.join(out_dir, mask_name))
    return

if __name__ == '__main__':
    do()