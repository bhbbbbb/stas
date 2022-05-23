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


class RandomResizedCropROI:
    def __init__(
        self,
        scale: Tuple[float, float] = (0.8, 2.0),
        output_len: int = 192,
    ) -> None:
        """Resize the input image to the given size.
        """
        self.scale = scale
        self.output_len = output_len
        return

    def crop_by_roi(self, img: Tensor, mask: Union[Tensor, None], roi: ROI
    ) -> Tuple[Tensor, Tensor]:

        # get the scale
        ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        # ratio = random.uniform(min(self.scale), max(self.scale))

        # scale the image 

        longer_edge = max(roi.width, roi.height)
        new_len = int(longer_edge / ratio)
        if new_len & 1:
            new_len += 1
        half_new_len = new_len // 2

        c_x, c_y = roi.center
        top, left = c_x - half_new_len, c_y - half_new_len
        top_to_pad = - min(0, top)
        left_to_pad = - min(0, left)
        top, left = max(0, top), max(0, left)

        img = TF.crop(img, top, left, new_len, new_len)
        # print(f'roi: {roi}')
        # print(f'new_len = {new_len}, top = {top}, left = {left}')
        if mask is not None:
            mask = TF.crop(mask, top, left, new_len, new_len)

        # pad the image
        if top_to_pad + left_to_pad:
            padding = [0, 0, top_to_pad, left_to_pad]
            img = TF.pad(img, padding, fill=0)
            # if mask is not None:
            #     mask = TF.pad(mask, padding, fill=0)
        img = TF.resize(img, (self.output_len, self.output_len), TF.InterpolationMode.BILINEAR)
        # if mask is not None:
        #     mask = TF.resize(mask, (LEN, LEN), TF.InterpolationMode.NEAREST)

        return img, mask
