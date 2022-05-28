from typing import OrderedDict
import torch
from torch import nn
from torch import Tensor
import numpy as np
from nfnets import NFNet # pylint: disable=import-error

# from .transform import ROI
# from transform import ROI

class Attention(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.head = head
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)
        self.proj = nn.Linear(dim, dim)

        return

    def forward(self, x: Tensor) -> Tensor:
        N, C = x.shape
        q: Tensor = self.q(x)
        q = q.reshape(N, self.head, C // self.head).permute(1, 0, 2)

        kv: Tensor = self.kv(x)
        k, v = kv.reshape(-1, 2, self.head, C // self.head).permute(1, 2, 0, 3)

        attn: Tensor = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(0, 1).reshape(N, C)
        x = self.proj(x)
        return x

class MyNfnet(NFNet):
    def __init__(self, **kwargs):
        num_classes = kwargs["num_classes"] 
        kwargs["num_classes"] = 1000
        super().__init__(**kwargs)
        self.fc = nn.Linear(1000, num_classes)
        return

    def forward(self, x):
        output = super().forward(x)
        return self.fc(output)

    @staticmethod
    def fix_output_layer(model_state_dict: OrderedDict, num_classes):

        fc_weight = np.random.normal(scale=0.01, size=[num_classes, 1000])
        fc_bias = np.random.normal(scale=0.1, size=[num_classes])

        model_state_dict["fc.weight"] = torch.tensor(fc_weight, dtype=torch.float32)
        model_state_dict["fc.bias"] = torch.tensor(fc_bias, dtype=torch.float32)
        model_state_dict.move_to_end("fc.weight", last=True)
        model_state_dict.move_to_end("fc.bias", last=True)
        return model_state_dict

class RoiAttentionNFnet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.nfnet = MyNfnet(**kwargs)
        self.attn = Attention(dim=3, head=3)
        return
    
    def forward(self, x: Tensor, rois: dict):
        c = x.shape[1]
        
        for idx, (roi_x, roi_y, roi_h, roi_w) in\
                            enumerate(zip(rois["x"], rois["y"], rois["height"], rois["width"])):
            x_roi = x[idx, :, roi_x:(roi_x + roi_h), roi_y:(roi_y + roi_w)]
            # x: 3 x h x w
            x_roi = x_roi.flatten(1).transpose(0, 1)
            # x: N x 3
            x_roi: Tensor = self.attn(x_roi)
            x_roi = x_roi.transpose(0, 1).reshape(c, roi_h, roi_w)
            
            x[idx, :, roi_x:(roi_x + roi_h), roi_y:(roi_y + roi_w)] = x_roi

        return self.nfnet(x)
    
    def load_pretrained_nfnet_weights(self, state_dict: OrderedDict):
        self.nfnet.load_state_dict(state_dict)
        return
    
    def exclude_from_weight_decay(self, name: str) -> bool:
        if name.startswith("nfnet"):
            return self.nfnet.exclude_from_weight_decay(name[len("nfnet."):])
        return False

    def exclude_from_clipping(self, name: str) -> bool:
        if name.startswith("nfnet"):
            return self.nfnet.exclude_from_clipping(name[len("nfnet."):])
        return False

# if __name__ == "__main__":
    # model = Attention(3, 3)
    # model = RoiAttentionNFnet(num_classes=2, stochdepth_rate=0.25)
#     for name, val in model.named_parameters():
#         print(name)
