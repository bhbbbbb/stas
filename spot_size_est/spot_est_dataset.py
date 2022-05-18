from typing import Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
from torch import Tensor
from torchvision import io
import pandas as pd
import numpy as np

from model_utils import BaseConfig

from semseg.augmentations import get_train_augmentation, get_val_augmentation
from stas.stas_dataset import StasDataset, DatasetConfig, M
from .bfs import BFS

IMAGE_SIZE = (471, 858)
IMAGE_SIZE_ = IMAGE_SIZE[0] * IMAGE_SIZE[1]

class SpotEstDatasetConfig(BaseConfig):
    SPOT_CSV: str

class SpotEstDataset(StasDataset):

    def __init__(
        self,
        config: DatasetConfig,
        split: Literal['train', 'val', 'inference'],
        test_dir: str = None
    ):

        if split == M.TRAIN:
            transform = get_train_augmentation(IMAGE_SIZE)
        else:
            transform = get_val_augmentation(IMAGE_SIZE)
        
        super().__init__(config, split, transform, test_dir)

        self.df = pd.read_csv(config.SPOT_CSV, dtype={'name': str})
        return
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = self.files[index]
        image = io.read_image(img_path)

        if self.mode == M.INFER:
            _, h, w = image.shape
            mask = torch.zeros([1, h, w]) # dummy mask
        else:
            mask_path = self.masks[index]
            mask = io.read_image(mask_path)
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        if self.mode == M.INFER:
            return image, self.names[index]

        if self.mode == M.TRAIN:
            size = BFS.get_smallest_white_spot(mask, 2).size
        else:
            # size = BFS.get_smallest_white_spot(mask, 2).size
            size = self.df['smallest_size'][int(self.names[index])] // 4
        return image, np.float32(size)
