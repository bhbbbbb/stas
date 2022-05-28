import random

from typing import Tuple, NamedTuple, Union

from torch import Tensor
from torchvision.transforms import functional as TF

from semseg.augmentations import (
    Compose,
    # RandomHorizontalFlip,
    # RandomVerticalFlip,
    Normalize
)


TRAIN_TRANSFORM = Compose([
        # RandomHorizontalFlip(p=0.5),
        # RandomVerticalFlip(p=0.5),
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


class Cropper:
    def __init__(
        self,
        scale: Tuple[float, float] = (0.5, 2.0),
        output_len: int = 192,
        max_roi_len: int = None,
    ) -> None:
        """Resize the input image to the given size.
        """
        self.scale = scale
        self.output_len = output_len
        self.max_roi_len = max_roi_len
        return

    @staticmethod
    def to_even(n: int):
        return n + 1 if n & 1 else n
    

    def random_scale_crop(
        self, img: Union[Tensor, None], mask: Union[Tensor, None], roi: ROI
    ):
        """crop w.r.t. roi, and leave the not-roi pixel in image.

        Args:
            img (Union[Tensor, None]): img to crop. to (output_len, output_len)
            mask (Union[Tensor, None]): mask.
            roi (ROI): roi.

        """

        o_h, o_w = img.shape[-2:]

        longer_roi_edge = max(roi.height, roi.width)
        max_scale_ratio = (self.max_roi_len / longer_roi_edge) - 0.05

        # get the scale
        if min(self.scale) > max_scale_ratio:
            ratio = max_scale_ratio
        else:
            ratio = random.uniform(min(self.scale), min(max(self.scale), max_scale_ratio))
        
        new_len = self.to_even(int(self.output_len / ratio))
        half_new_len = new_len // 2

        # crop image and mask
        c_x, c_y = roi.center
        top, left = c_x - half_new_len, c_y - half_new_len
        bottom, right = c_x + half_new_len, c_y + half_new_len

        roi_x = max(roi.x, top)
        roi_y = max(roi.y, left)
        roi_h = min(roi.x + new_len, roi.x + roi.height) - roi_x
        roi_w = min(roi.y + new_len, roi.y + roi.width) - roi_y
        roi_x -= top
        roi_y -= left

        def fix(a):
            return int(a * ratio)

        roi_x, roi_y, roi_h, roi_w = fix(roi_x), fix(roi_y), fix(roi_h), fix(roi_w)

        top_pad, left_pad, bottom_pad, right_pad = 0, 0, 0, 0
        if top < 0:
            top_pad = - top
            top = 0
        if left < 0:
            left_pad = - left
            left = 0
        if bottom > o_h: 
            bottom_pad = bottom - o_h
            bottom = o_h
        if right > o_w:
            right_pad = right - o_w
            right = o_w

        h, w = bottom - top, right - left
        if img is not None:
            img = TF.crop(img, top, left, h, w)

            # pad the image
            if left_pad + top_pad + bottom_pad + right_pad > 0:
                padding = [left_pad, top_pad, right_pad, bottom_pad]
                img = TF.pad(img, padding, fill=0)

            img = TF.resize(img, (self.output_len, self.output_len), TF.InterpolationMode.BILINEAR)
        
        if mask is not None:
            mask = TF.crop(
                mask,
                roi.x,
                roi.y,
                roi.height,
                roi.width,
            )

        return img, ROI(mask.sum().item(), roi_x, roi_y, roi_h, roi_w)

    def crop_fix_by_roi(
        self, img: Union[Tensor, None], mask: Union[Tensor, None], roi: ROI
    ) -> Tuple[Tensor, Tensor]:
        if img is not None:
            img = TF.resized_crop(
                img,
                roi.x,
                roi.y,
                roi.height,
                roi.width,
                (self.output_len, self.output_len)
            )
        if mask is not None:
            mask = TF.crop(
                mask,
                roi.x,
                roi.y,
                roi.height,
                roi.width,
                # (self.output_len, self.output_len)
            )
        return img, mask
    
    def crop_by_roi(self, img: Union[Tensor, None], mask: Union[Tensor, None], roi: ROI
    ) -> Tuple[Tensor, Tensor]:
        """crop and remove everything except roi pixels"""

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
