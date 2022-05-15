from typing import Tuple
from torch import Tensor
from torch import nn
from semseg.losses import OhemCrossEntropy

class LossConfig:
    initial_cls_weights: Tuple
    cls_weights: Tuple
    loss_warmup_epochs: int
    ignore_label: int
    device: str

class WarmupOhemCrossEntropy(nn.Module):
    """
    With warmup version
    """

    def __init__(self, config: LossConfig, start_epoch: int):
        super().__init__()
        def to_ratio(w1, w2):
            return w1 / (w1 + w2)
        self.config = config
        init_cls_weights = to_ratio(*config.initial_cls_weights)
        self.final_cls_weights = to_ratio(*config.cls_weights)
        self.decay_ratio = (self.final_cls_weights - init_cls_weights) / config.loss_warmup_epochs
        assert self.decay_ratio != 0
        self.weights = init_cls_weights
        self.criterion = None
        self._stepn(start_epoch)
        if self.criterion is None: # equivalent to init_w == final_w
            w = Tensor([self.weights, (1 - self.weights)]).to(config.device)
            self.criterion = OhemCrossEntropy(self.config.ignore_label, w)

        return
    
    def get_weights(self):
        return (self.weights, 1 - self.weights)
    
    def _stepn(self, n: int):
        if self.weights == self.final_cls_weights:
            return
        self.weights += self.decay_ratio * n
        if (self.weights - self.final_cls_weights) / self.decay_ratio > 0:
            self.weights = self.final_cls_weights
        w = Tensor([self.weights, (1 - self.weights)]).to(self.config.device)
        self.criterion = OhemCrossEntropy(self.config.ignore_label, w)
        return

    def step(self):
        self._stepn(1)
        return

    def forward(self, *args):
        return self.criterion(*args)
    