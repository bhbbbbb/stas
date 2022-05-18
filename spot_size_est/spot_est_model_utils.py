import os
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler
from torch import nn
# from torch.nn import functional as F
from torch import Tensor
import torch
import numpy as np
import pandas as pd

from model_utils import BaseModelUtils
from model_utils import Loss as BaseLoss, Criteria
# from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from stas.config import Config
from .spot_est_dataset import SpotEstDataset
from .model import SpotEst
from .loss import L1L2Loss

Loss = Criteria.register_criterion(
    short_name='spot_loss',
    full_name='Loss',
    primary=True,
    plot=True
)(BaseLoss)

class SpotEstModelUtils(BaseModelUtils):
    model: SpotEst
    config: Config

    scaler: GradScaler
    loss_fn: L1L2Loss
    # dice: LossDice
    # scaler: GradScaler
    scheduler: _LRScheduler

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        optimizer,
        scheduler,
        start_epoch: int,
        root: str,
        history_utils,
        logger
    ):
        super().__init__(
            model, config, optimizer, scheduler, start_epoch, root, history_utils, logger)
        self.scaler = GradScaler(enabled=config.AMP)
        # self.loss_fn = get_loss(config.loss_name, config.ignore_label, config.cls_weights)
        self.loss_fn = L1L2Loss()
        # self.dice = LossDice()
        return


    @classmethod
    def start_new_training_(cls, model: SpotEst, config: Config, pretrained_path: str):
        model.load_pretrained(pretrained_path)
        return super().start_new_training(model, config)
    
    @staticmethod
    def _get_optimizer(model: nn.Module, config: Config):
        return get_optimizer(
            model,
            config.optimizer_name,
            config.learning_rate,
            config.weight_decay,
        )
    
    @staticmethod
    def _get_scheduler(optimizer, config: Config, state_dict: dict):
        scheduler = get_scheduler(
            config.scheduler_name,
            optimizer,
            config.max_iters,
            config.power,
            config.warmup_iters,
            config.warmup_ratio,
        )
        if state_dict is not None:
            scheduler.load_state_dict(state_dict)
        return scheduler

    def _train_epoch(self, train_dataset: SpotEstDataset) -> Criteria:
        
        self.model.train()

        train_loss = 0.0
        idx = 0
        pbar = tqdm(train_dataset.dataloader)
        for img, sizes in pbar:
            img: Tensor
            sizes: Tensor
            self.optimizer.zero_grad(set_to_none=True)

            img = img.to(self.config.device)
            sizes = sizes.to(self.config.device)
            
            with autocast(enabled=self.config.AMP):
                preds: Tensor = self.model(img)
                loss: Tensor = self.loss_fn(preds, sizes)
                loss = loss.float()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            torch.cuda.synchronize()

            lr = self.scheduler.get_lr()
            lr = sum(lr) / len(lr)
            running_loss = loss.item()
            train_loss += running_loss

            pbar.set_description(
                f'LR: {lr:.1e} Loss: {running_loss:.4e}| {int(preds[0])}, {int(sizes[0])}'
            )
            idx += 1
        
        train_loss /= idx
        torch.cuda.empty_cache()
        return Criteria(Loss(train_loss))

    
    @torch.no_grad()
    def _eval_epoch(self, eval_dataset: SpotEstDataset) -> Criteria:
        self.model.eval()

        step = 0
        total_loss = 0.0
        for images, sizes in tqdm(eval_dataset.dataloader):
            step += 1
            images: Tensor = images.to(self.config.device)
            sizes: Tensor = sizes.to(self.config.device)
            preds: Tensor = self.model(images)
            loss: Tensor = self.loss_fn(preds, sizes)
            total_loss += loss.item()
        

        total_loss /= step

        return Criteria(Loss(total_loss))
    
    @torch.inference_mode()
    def inference(self, test_dataset: SpotEstDataset, out_dir: str):
        self.model.eval()

        name_list = np.empty([0], dtype=str)
        pred_list = np.empty([0], dtype=int)
        for images, names in tqdm(test_dataset.dataloader):
            images: Tensor = images.to(self.config.device)
            names: Tensor
            preds: Tensor = self.model(images)
            # preds *= IMAGE_SIZE_
            preds = preds.cpu().to(torch.uint8)
            name_list = np.concatenate([name_list, names])
            pred_list = np.concatenate([pred_list, preds.numpy()])
        df = pd.DataFrame(
            {
                'name': name_list,
                'pred_size': pred_list,
            }
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'size_pred.csv')
        df.to_csv(out_path)
        return
