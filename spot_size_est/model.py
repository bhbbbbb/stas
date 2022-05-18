from typing import Tuple, OrderedDict
import math

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads.segformer import MLP, ConvModule

class SpotEst(BaseModel):
    def __init__(self, num_classes: int):
        # assert image_size == (471, 858)
        image_size = (471, 858)
        super().__init__(num_classes=num_classes)
        dims = self.backbone.channels
        embed_dim = 256
        self.linear_fuse1 = ConvModule(embed_dim*4, embed_dim)
        self.linear_fuse2 = ConvModule(embed_dim, num_classes)
        for i, dim in enumerate(dims):
            self.add_module(f'linear_c{i+1}', MLP(dim, embed_dim))
        
        d = num_classes * math.ceil(image_size[0] / 32) * math.ceil(image_size[1] / 32)
        d_ = d // num_classes
        self.fc1 = nn.Linear(d, d_)
        self.fc2 = nn.Linear(d_, 1)
        self.apply(self._init_weights)
        return
    
    def forward(self, x):
        features: Tuple[Tensor, Tensor, Tensor, Tensor] = self.backbone(x)
        B, _, H, W = features[-1].shape
        outs = []

        for i, feature in enumerate(features[:-1]):
            # pylint: disable=eval-used
            cf = eval(f'self.linear_c{i+1}')\
                (feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            tem = F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(tem)

        tem = self.linear_c4(features[-1]).permute(0, 2, 1).reshape(B, -1, *features[-1].shape[-2:])
        outs.append(tem)
        
        tem = torch.cat(outs[::-1], dim=1)
        tem: Tensor = self.linear_fuse2(self.linear_fuse1(tem))
        tem = tem.flatten(1)
        tem = F.relu(self.fc1(tem), inplace=True)
        tem = F.relu(self.fc2(tem), inplace=True)
        return tem.squeeze(1)

    def load_pretrained(self, path: str):
        d: dict = torch.load(path)['model_state_dict']
        new_dict = OrderedDict()
        for key, value in d.items():
            key: str
            if key.startswith('decode_head'):
                break
            key = key[len('backbone.'):]
            new_dict[key] = value
        self.backbone.load_state_dict(new_dict)
        return
