import math
import os
import json
from typing import Dict, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import random

import torch 
# from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, Dataset
from torchvision import io
# from torchvision.transforms import functional as TF

from sklearn.utils import shuffle

# from semseg.augmentations import get_train_augmentation, get_val_augmentation
from model_utils.base import BaseConfig
# from tqdm import tqdm

from .transform import Cropper, TRAIN_TRANSFORM, VALID_TRANSFORM, ROI

class M:
    """enum"""
    INFER = 'inference'
    TRAIN = 'train'
    VALID = 'val'

class DatasetConfig(BaseConfig):
    IMGS_ROOT: str
    MASK_ROOT: str
    ROIS_JSON_PATH: str
    # VALID_ROI_JSON_PATH: str
    TRAIN_SPLIT: str
    VALID_SPLIT: str
    batch_size: Dict[str, int]
    num_workers: int
    drop_last: bool
    pin_memory: bool
    small_spot_threshold: int = 200
    nf_variant: str
    nf_batch_size_inf: int
    train_img_len: int = 192
    max_roi_len_after_crop: int = 100
    # max_roi_len: int = 10000
    # min_roi_len: int = 100
    max_roi_size: int
    min_roi_size: int

    random_drop_black_rate: float = 0.7


    @property
    def persistent_workers(self):
        return self.num_workers > 0 and os.name == 'nt'

def get_files(config: DatasetConfig, mode: str):
    assert mode in [M.TRAIN, M.VALID]

    def do(split_path: str):
        with open(split_path, 'r', encoding='utf-8') as fin:
            names: list = json.load(fin)
            imgs = [os.path.join(config.IMGS_ROOT, name + '.jpg') for name in names]
            masks = [os.path.join(config.MASK_ROOT, name + '.png') for name in names]
            # rois = [os.path.join(config.ROIS_ROOT, name + '.json') for name in names]
        return names, imgs, masks
    if mode == M.TRAIN:
        return do(config.TRAIN_SPLIT)
    
    return do(config.VALID_SPLIT)


class SingleImageSpotDataset(Dataset):
    """For inference only"""

    def __init__(
        self,
        img_path: str,
        rois: List[ROI],
        config: DatasetConfig,
    ):
        self.img = io.read_image(img_path)
        h, w = self.img.shape[-2:]
        dummy_mask = torch.zeros([1, h, w])
        self.img, _ = VALID_TRANSFORM(self.img, dummy_mask)
        self.cropper = Cropper(
            (1.0, 1.0),
            output_len=config.train_img_len,
            max_roi_len=config.max_roi_len_after_crop,
        )
        self.rois = rois
        # self.roi_masks = roi_masks
        self.config = config
        # print(f'Found {len(rois)} rois.')
        return
    
    def __len__(self):
        return len(self.rois)
    
    def __getitem__(self, index: int):
        roi = self.rois[index]
        img, _ = self.cropper.crop_fix_by_roi(self.img, None, roi)
        return img, roi.size, index

    @property
    def dataloader(self):
        return DataLoader(
            self,
            batch_size=self.config.nf_batch_size_inf,
            num_workers=0,
            pin_memory=self.config.pin_memory,
        )


class SingleImageRoiSpotDataset(SingleImageSpotDataset):
    """For inference only"""
    def __getitem__(self, index: int):
        roi = self.rois[index]
        img, roi = self.cropper.random_scale_crop(self.img, None, roi)
        return img, roi._asdict(), index

class SpotDataset(IterableDataset):
    # pylint: disable=abstract-method
    
    names: List[str]
    imgs: List[str]
    masks: List[str]
    rois: List[List[dict]]

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
        self.roi_transform = Cropper(
            scale=((0.8, 2.0) if mode == M.TRAIN else (1.0, 1.0)),
            output_len=config.train_img_len,
            max_roi_len=config.max_roi_len_after_crop,
        )
        
        if transform is None:
            if mode == M.TRAIN:
                self.transform = TRAIN_TRANSFORM
            else:
                self.transform = VALID_TRANSFORM

        self.names, self.imgs, self.masks = get_files(config, mode)
        
        with open(config.ROIS_JSON_PATH, 'r', encoding='utf8') as fin:
            self.rois = json.load(fin)

        print(f'Found {len(self.imgs)} {mode} images.')
        return

    # def __len__(self) -> int:
    #     return self.config.spot_iters_per_epoch
    
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
        
        
        for roi in self.rois[int(self.names[index])]:
            roi = ROI(**roi)
            # if (
            #     (roi.height > self.config.max_roi_len or roi.width > self.config.max_roi_len)
            #     or
            #     (roi.height < self.config.min_roi_len or roi.width < self.config.min_roi_len)
            # ):
            #     continue
            if (
                roi.size > self.config.max_roi_size or roi.size < self.config.min_roi_size
            ):
                continue
            cropped_img, cropped_mask = self.roi_transform.crop_fix_by_roi(image, mask, roi)
            num_white = cropped_mask.sum().item()
            if 0 < num_white < self.config.small_spot_threshold:
                continue
            if (
                self.mode == M.TRAIN and num_white == 0
                and random.random() < self.config.random_drop_black_rate
            ):
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


class RoiSpotDataset(SpotDataset):
    """for RoiAttentionNFnet"""
    # pylint: disable=abstract-method
    
    names: List[str]
    imgs: List[str]
    masks: List[str]
    rois: List[List[dict]]

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
            for img, roi in self.get_item(idx):
                yield img, roi._asdict()
    
    def get_item(self, index: int):
        img_path = self.imgs[index]
        image = io.read_image(img_path)

        mask_path = self.masks[index]
        mask = io.read_image(mask_path)
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        
        for roi in self.rois[int(self.names[index])]:
            roi = ROI(**roi)
            cropped_img, roi = self.roi_transform.random_scale_crop(image, mask, roi)
            num_white = roi.size
            if 0 < num_white < self.config.small_spot_threshold:
                continue
            if (
                self.mode == M.TRAIN and num_white == 0
                and random.random() < self.config.random_drop_black_rate
            ):
                continue

            yield cropped_img, roi
