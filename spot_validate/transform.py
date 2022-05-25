import random

from typing import Tuple, NamedTuple, Union

from torch import Tensor
from torchvision.transforms import functional as TF

from semseg.augmentations import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Normalize
)


TRAIN_TRANSFORM = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        # RandomGaussianBlur((3, 3), p=0.5),
        # RandomGrayscale(p=0.5),
        # RandomRotation(degrees=10, p=0.3, seg_fill=seg_fill),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
VALID_TRANSFORM = Compose([
        # RandomHorizontalFlip(p=0.5),
        # RandomVerticalFlip(p=0.5),
        # RandomGaussianBlur((3, 3), p=0.5),
        # RandomGrayscale(p=0.5),
        # RandomRotation(degrees=10, p=0.3, seg_fill=seg_fill),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

class ROI(NamedTuple):
    size: int
    x: int
    y: int
    height: int
    width: int

    @property
    def center(self):
        return (self.x + (self.height // 2), self.y + (self.width // 2))

    @property
    def bottom(self):
        return self.x + self.height

    @property
    def right(self):
        return self.y + self.width


class RandomResizedCropROI:
    def __init__(
        self,
        scale: Tuple[float, float] = (0.5, 2.0),
        output_len: int = 192,
    ) -> None:
        """Resize the input image to the given size.
        """
        self.scale = scale
        self.output_len = output_len
        return

    @staticmethod
    def to_even(n: int):
        return n + 1 if n & 1 else n
    
    def crop_by_roi(self, img: Union[Tensor, None], mask: Union[Tensor, None], roi: ROI
    ) -> Tuple[Tensor, Tensor]:

        # get the scale
        ratio = random.uniform(min(self.scale), max(self.scale))

        new_len = self.to_even(int(self.output_len / ratio))

        h = new_len if roi.height > new_len else self.to_even(roi.height)
        w = new_len if roi.width  > new_len else self.to_even(roi.width)
        # crop image and mask
        c_x, c_y = roi.center
        top, left = c_x - (h // 2), c_y - (w // 2)
        # bottom, right = c_x + (h // 2), c_y + (w // 2)

        if img is not None:
            # img = img[top: bottom, left: right].detach().clone()
            img = TF.crop(img, top, left, h, w)

            # pad the image
            v_pad = (new_len - h) // 2
            h_pad = (new_len - w) // 2
            if v_pad + h_pad:
                padding = [h_pad, v_pad, h_pad, v_pad]
                img = TF.pad(img, padding, fill=0)

            img = TF.resize(img, (self.output_len, self.output_len), TF.InterpolationMode.BILINEAR)
        
        if mask is not None:
            # mask = mask[top: bottom, left: right].detach().clone()
            mask = TF.crop(mask, top, left, h, w)

        return img, mask
