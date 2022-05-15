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
from semseg.augmentations import get_train_augmentation, get_val_augmentation
from model_utils.base import BaseConfig
from tqdm import tqdm

class DatasetConfig(BaseConfig):
    num_classes: int
    img_size: Tuple = (471, 858)
    val_size: Tuple = (942, 1716)
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
        assert mode in ['train', 'val']

        def do(split_path: str):
            with open(split_path, 'r', encoding='utf-8') as fin:
                names = json.load(fin)
                imgs = [os.path.join(self.IMGS_ROOT, name + '.jpg') for name in names]
                masks = [os.path.join(self.MASK_ROOT, name + '.png') for name in names]
            return names, imgs, masks
        if mode == 'train':
            return do(self.TRAIN_SPLIT)
        
        return do(self.VALID_SPLIT)


class StasDataset(Dataset):
    
    def __init__(
        self,
        config: DatasetConfig,
        split: Literal['train', 'val', 'inference'] = 'train',
        transform = None,
    ):

        super().__init__()
        assert split in ['train', 'val', 'inference']
        self.mode = split
        self.config = config
        self.transform = transform
        
        if transform is None:
            if split == 'train':
                self.transform = get_train_augmentation(config.img_size)
            else:
                self.transform = get_val_augmentation(config.val_size)

        self.names, self.files, self.masks =\
            config.get_files('train' if split == 'train' else 'val')
    
        print(f'Found {len(self.files)} {split} images.')
        return

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path, mask_path = self.files[index], self.masks[index]

        image = io.read_image(img_path)
        label = io.read_image(mask_path)
        
        if self.transform:
            image, label = self.transform(image, label)
        
        if self.mode == 'inference':
            return image, self.encode(label), self.names[index]
        return image, self.encode(label)
        # return image, self.encode(label.squeeze().numpy()).long()

    @property
    def dataloader(self):
        batch_size = self.config.batch_size['train' if self.mode == 'train' else 'val']
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=(self.mode == 'train'),
            num_workers=min(self.config.num_workers, batch_size),
            persistent_workers=self.config.persistent_workers,
            drop_last=self.config.drop_last,
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
                self.config.val_size,
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
