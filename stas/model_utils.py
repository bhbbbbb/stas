from tqdm import tqdm
from model_utils import BaseModelUtils
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler
from torch import nn
from torch import Tensor
import torch
from semseg.models import SegFormer
from semseg.metrics import Metrics
# from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from .config import Config
from .criteria import *
from .stas_dataset import StasDataset
from .loss import WarmupOhemCrossEntropy


class SemsegModelUtils(BaseModelUtils):
    model: SegFormer
    config: Config

    scaler: GradScaler
    loss_fn: WarmupOhemCrossEntropy
    # dice: LossDice
    scaler: GradScaler
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
        self.loss_fn = WarmupOhemCrossEntropy(config, start_epoch)
        # self.dice = LossDice()
        return


    @classmethod
    def start_new_training(cls, model: SegFormer, config: Config):
        model.init_pretrained(config.pretrained)
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

    def _train_epoch(self, train_dataset: StasDataset) -> Criteria:
        
        self.model.train()

        train_loss = 0.0
        idx = 0
        print(f'cls_weights = {self.loss_fn.get_weights()}')
        pbar = tqdm(train_dataset.dataloader)
        for img, lbl in pbar:
            img: Tensor
            lbl: Tensor
            self.optimizer.zero_grad(set_to_none=True)

            img = img.to(self.config.device)
            lbl = lbl.to(self.config.device)
            
            with autocast(enabled=self.config.AMP):
                logits: Tensor = self.model(img)
                loss: Tensor = self.loss_fn(logits, lbl)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            torch.cuda.synchronize()

            lr = self.scheduler.get_lr()
            lr = sum(lr) / len(lr)
            running_loss = loss.item()
            train_loss += running_loss

            pbar.set_description(f"LR: {lr:.4e} Running Loss: {running_loss:.6f}")
            idx += 1
        
        self.loss_fn.step()
        train_loss /= idx
        # writer.add_scalar('train/loss', train_loss, epoch)
        torch.cuda.empty_cache()
        return Criteria(Loss(train_loss))

    
    @torch.no_grad()
    def _eval_epoch(self, eval_dataset: StasDataset) -> Criteria:
        self.model.eval()
        metrics = Metrics(self.config.num_classes, self.config.ignore_label, self.config.device)

        # dice_score = 0.0
        step = 0
        for images, labels in tqdm(eval_dataset.dataloader):
            step += 1
            images: Tensor = images.to(self.config.device)
            labels: Tensor = labels.to(self.config.device)
            preds: Tensor = self.model(images)
            preds = preds.softmax(dim=1)
            # tem: Tensor = self.dice(preds, labels)
            # dice_score += tem.item()
            metrics.update(preds, labels)
        
        # dice_score /= step

        def d100(a: list, b):
            return (a[0] / 100, a[1] / 100), b / 100
        
        ious, miou = metrics.compute_iou()
        acc, macc = metrics.compute_pixel_acc()
        f1, mf1 = metrics.compute_f1()

        (bg_iou, tgt_iou), miou = d100(ious, miou)
        (bg_acc, tgt_acc), macc = d100(acc, macc)
        (bg_f1, tgt_f1), mf1 = d100(f1, mf1)

        return Criteria(
            # Dice(dice_score),
            TargetIou(tgt_iou),
            BackgroundIou(bg_iou),
            # MIou(miou),
            TargetAccuracy(tgt_acc),
            BackgroundAccuracy(bg_acc),
            # MAccuracy(macc),
            TargetF1(tgt_f1),
            BackgroundF1(bg_f1),
            # MF1(mf1),
        )
    
    @torch.inference_mode()
    def splash(self, test_dataset: StasDataset, out_dir: str = 'splash',
                    num_of_output: int = 1):
        self.model.eval()

        idx = 0
        for images, labels, names in tqdm(test_dataset.dataloader):
            images: Tensor = images.to(self.config.device)
            labels: Tensor = labels.to(self.config.device)
            preds: Tensor = self.model(images).softmax(dim=1)
            preds = preds.argmax(dim=1).cpu().to(torch.uint8)

            test_dataset.splash_to_file(preds, names, out_dir)
            idx += 1
            if idx == num_of_output:
                return

