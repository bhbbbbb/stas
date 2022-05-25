import os
import json
from typing import Tuple, Dict
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch 
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import io
from torchvision.transforms import functional as TF
from semseg.augmentations import get_val_augmentation
from semseg.augmentations import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Normalize,
)
from model_utils.base import BaseConfig
from tqdm import tqdm

class M:
    """enum"""
    INFER = 'inference'
    TRAIN = 'train'
    VALID = 'val'

class DatasetConfig(BaseConfig):
    num_classes: int
    train_size: Tuple
    val_size: Tuple
    inf_size: Tuple
    IMGS_ROOT: str
    MASK_ROOT: str
    TRAIN_SPLIT: str
    VALID_SPLIT: str
    batch_size: Dict[str, int]
    num_workers: int
    drop_last: bool
    pin_memory: bool

    @property
    def persistent_workers(self):
        return self.num_workers > 0 and os.name == 'nt'

    def get_files(self, mode: str):
        assert mode in [M.TRAIN, M.VALID]

        def do(split_path: str):
            with open(split_path, 'r', encoding='utf-8') as fin:
                names = json.load(fin)
                imgs = [os.path.join(self.IMGS_ROOT, name + '.jpg') for name in names]
                masks = [os.path.join(self.MASK_ROOT, name + '.png') for name in names]
            return names, imgs, masks
        if mode == M.TRAIN:
            return do(self.TRAIN_SPLIT)
        
        return do(self.VALID_SPLIT)


class StasDataset(Dataset):
    
    def __init__(
        self,
        config: DatasetConfig,
        split: Literal['train', 'val', 'inference'] = M.TRAIN,
        transform = None,
        test_dir: str = None,
    ):

        super().__init__()
        assert split in [M.TRAIN, M.VALID, M.INFER]
        self.mode = split
        self.config = config
        self.transform = transform
        
        if transform is None:
            if split == M.TRAIN:
                mid_scale = config.inf_size[0] / config.train_size[0]
                self.transform = Compose([
                    # ColorJitter(brightness=0.0, contrast=0.5, saturation=0.5, hue=0.5),
                    # RandomAdjustSharpness(sharpness_factor=0.1, p=0.5),
                    # RandomAutoContrast(p=0.2),
                    RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    # RandomGaussianBlur((3, 3), p=0.5),
                    # RandomGrayscale(p=0.5),
                    # RandomRotation(degrees=10, p=0.3, seg_fill=seg_fill),
                    RandomResizedCrop(
                        config.train_size,
                        scale=(mid_scale - 0.5, mid_scale + 0.5),
                        seg_fill=0,
                    ),
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
            else:
                self.transform = get_val_augmentation(
                    config.val_size if split == M.VALID else config.inf_size
                    # config.val_size
                )

        if test_dir is not None:
            assert self.mode == M.INFER
            assert os.path.isdir(test_dir)
            filenames = os.listdir(test_dir)
            self.names = [os.path.splitext(filename)[0] for filename in filenames]
            self.files = [os.path.join(test_dir, filename) for filename in filenames]
            self.masks = None

        else:
            self.names, self.files, self.masks =\
                config.get_files(M.TRAIN if split == M.TRAIN else M.VALID)
    
        print(f'Found {len(self.files)} {split} images.')
        return

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = self.files[index]
        image = io.read_image(img_path)

        if self.mode == M.INFER:
            _, h, w = image.shape
            label = torch.zeros([1, h, w]) # dummy mask
        else:
            mask_path = self.masks[index]
            label = io.read_image(mask_path)
        
        if self.transform:
            image, label = self.transform(image, label)
        
        if self.mode == M.INFER:
            return image, self.names[index]
        return image, self.encode(label)
        # return image, self.encode(label.squeeze().numpy()).long()

    @property
    def dataloader(self):
        batch_size = self.config.batch_size[M.TRAIN if self.mode == M.TRAIN else M.VALID]
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=(self.mode == M.TRAIN),
            num_workers=min(self.config.num_workers, batch_size),
            persistent_workers=self.config.persistent_workers,
            drop_last=(self.mode != M.INFER),
            pin_memory=self.config.pin_memory,
        )

    @staticmethod
    def encode(label: Tensor) -> Tensor:
        return label.squeeze().long()
        # label = self.label_map[label]
        # return torch.from_numpy(label)


    def splash_to_file(self, labels: Tensor, filenames: str, output_dir: str):
        # labels: b x h x w
        labels *= 255
        _, h, w = labels.shape
        os.makedirs(output_dir, exist_ok=True)
        for filename, label in zip(filenames, labels):
            label = label.view(1, h, w)
            label = TF.resize(
                label,
                self.config.inf_size,
                interpolation=TF.InterpolationMode.NEAREST
            )
            filename += '.png'
            filename = os.path.join(output_dir, filename)
            io.write_png(label, filename)
        return

def get_labels_ratio(dataset: StasDataset):
    num_classes = dataset.config.num_classes
    counts = torch.zeros(num_classes, dtype=torch.long)

    for _, labels in tqdm(dataset.dataloader):
        labels: Tensor
        labels = labels.view(-1)
        counts += labels.bincount(minlength=num_classes)
    max_count = counts.max()
    return counts / max_count
