import math
import os
import json
from typing import Dict, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch 
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, Dataset
from torchvision import io
# from torchvision.transforms import functional as TF

from sklearn.utils import shuffle

# from semseg.augmentations import get_train_augmentation, get_val_augmentation
from model_utils.base import BaseConfig
# from tqdm import tqdm

from .transform import RandomResizedCropROI, TRAIN_TRANSFORM, VALID_TRANSFORM, ROI

class M:
    """enum"""
    INFER = 'inference'
    TRAIN = 'train'
    VALID = 'val'

class DatasetConfig(BaseConfig):
    IMGS_ROOT: str
    MASK_ROOT: str
    ROIS_ROOT: str
    TRAIN_SPLIT: str
    VALID_SPLIT: str
    batch_size: Dict[str, int]
    num_workers: int
    drop_last: bool
    pin_memory: bool
    small_spot_threshold: int = 200
    nf_variant: str
    nf_batch_size_inf: int

    @property
    def persistent_workers(self):
        return self.num_workers > 0 and os.name == 'nt'

LEN = {
    'F0': 192,
    'F1': 224,
}

def get_files(config: DatasetConfig, mode: str):
    assert mode in [M.TRAIN, M.VALID]

    def do(split_path: str):
        with open(split_path, 'r', encoding='utf-8') as fin:
            names: list = json.load(fin)
            imgs = [os.path.join(config.IMGS_ROOT, name + '.jpg') for name in names]
            masks = [os.path.join(config.MASK_ROOT, name + '.png') for name in names]
            rois = [os.path.join(config.ROIS_ROOT, name + '.json') for name in names]
        return names, imgs, masks, rois
    if mode == M.TRAIN:
        return do(config.TRAIN_SPLIT)
    
    return do(config.VALID_SPLIT)


class SingleImageSpotDataset(Dataset):
    """For inference only"""

    def __init__(
        self,
        img_path: str,
        rois: List[ROI],
        roi_masks: List[Tensor],
        config: DatasetConfig,
    ):
        self.img = io.read_image(img_path)
        h, w = self.img.shape[-2:]
        dummy_mask = torch.zeros([1, h, w])
        self.img, _ = VALID_TRANSFORM(self.img, dummy_mask)
        self.cropper = RandomResizedCropROI((1.0, 1.0))
        self.rois = rois
        self.roi_masks = roi_masks
        self.config = config
        # print(f'Found {len(rois)} rois.')
        return
    
    def __len__(self):
        return len(self.rois)
    
    def __getitem__(self, index: int):
        roi = self.rois[index]
        r_mask = self.roi_masks[index]
        img = self.cropper.crop_by_roi_mask(self.img, r_mask, roi)
        return img, roi.size, index

    @property
    def dataloader(self):
        return DataLoader(
            self,
            batch_size=self.config.nf_batch_size_inf,
            num_workers=0,
            pin_memory=self.config.pin_memory,
        )


class SpotDataset(IterableDataset):
    # pylint: disable=abstract-method
    
    names: list
    imgs: list
    masks: list
    rois: list

    def __init__(
        self,
        config: DatasetConfig,
        mode: Literal['train', 'val'] = M.TRAIN,
        transform = None,
    ):

        super().__init__()
        assert mode in [M.TRAIN, M.VALID]
        self.mode = mode
        self.config = config
        self.transform = transform
        self.roi_transform = RandomResizedCropROI(
            (0.5, 2.0) if M.TRAIN else (1.0, 1.0), LEN[self.config.nf_variant]
        )
        
        if transform is None:
            if mode == M.TRAIN:
                self.transform = TRAIN_TRANSFORM
            else:
                self.transform = VALID_TRANSFORM

        self.names, self.imgs, self.masks, self.rois = get_files(config, mode)
        print(f'Found {len(self.imgs)} {mode} images.')
        return

    # def __len__(self) -> int:
    #     return len(self.imgs)
    
    def __iter__(self):
        worker_info = get_worker_info()
        start = 0
        end = len(self.imgs)
        if self.mode == M.TRAIN:
            indices = shuffle(range(end))
        else:
            indices = range(end)
        
        if worker_info is not None:
            len_per_worker = math.ceil(end // worker_info.num_workers)
            start = len_per_worker * worker_info.id
            end = start + len_per_worker
            end = min(len(self.imgs), end)

        for idx in indices[start:end]:
            for img, num_white in self.get_item(idx):
                yield img, num_white
    
    def get_item(self, index: int):
        img_path = self.imgs[index]
        image = io.read_image(img_path)

        mask_path = self.masks[index]
        mask = io.read_image(mask_path)
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        with open(self.rois[index], 'r', encoding='utf-8') as fin:
            rois = json.load(fin)
        for roi in rois:
            roi = ROI(**roi)
            cropped_img, cropped_mask = self.roi_transform.crop_by_roi(image, mask, roi)
            num_white = cropped_mask.sum().item()
            if 0 < num_white < self.config.small_spot_threshold:
                continue
            yield cropped_img, num_white

    @property
    def dataloader(self):
        batch_size = self.config.batch_size[M.TRAIN if self.mode == M.TRAIN else M.VALID]
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=min(self.config.num_workers, batch_size),
            persistent_workers=self.config.persistent_workers,
            drop_last=(self.mode == M.TRAIN),
            pin_memory=self.config.pin_memory,
        )
