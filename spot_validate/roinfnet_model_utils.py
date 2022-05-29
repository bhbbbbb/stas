from typing import List, Tuple, Union
import torch
from torch import Tensor
from torch import nn
from torch.cuda import amp

from tqdm import tqdm

from model_utils import BaseModelUtils, Criteria
from nfnets import SGD_AGC # pylint: disable=import-error

from .dataset import SpotDataset, SingleImageRoiSpotDataset 
from .model import RoiAttentionNFnet
from .config import NfnetConfig
from .transform import ROI #, RandomResizedCropROI, VALID_TRANSFORM
from .metrics import Metrics
from .criteria import (
    Loss,
    Accuracy,
    Precision,
    Recall,
    FScore,
    FPVal,
)


class RoiNfnetModelUtils(BaseModelUtils):
    config: NfnetConfig

    def __init__(
        self,
        model: nn.Module,
        config: NfnetConfig,
        optimizer,
        scheduler,
        start_epoch: int,
        root: str,
        history_utils,
        logger,
    ):
        self.scaler = amp.GradScaler()
        cls_weights = Tensor(config.nf_cls_weight).to(config.device)
        self.criterion = nn.CrossEntropyLoss(weight=cls_weights)
        super().__init__(
            model,
            config,
            optimizer,
            scheduler,
            start_epoch,
            root,
            history_utils,
            logger
        )
    
    @staticmethod
    def _get_optimizer(model: RoiAttentionNFnet, config: NfnetConfig):

        optimizer = SGD_AGC(
            # The optimizer needs all parameter names 
            # to filter them by hand later
            named_params=model.named_parameters(), 
            lr=config.learning_rate,
            momentum=config.momentum,
            clipping=config.clipping,
            weight_decay=config.weight_decay, 
            nesterov=config.nesterov,
        )
        # Find desired parameters and exclude them 
        # from weight decay and clipping
        for group in optimizer.param_groups:
            name = group["name"] 
            
            if model.exclude_from_weight_decay(name):
                group["weight_decay"] = 0

            if model.exclude_from_clipping(name):
                group["clipping"] = None
        return optimizer
    
    @staticmethod
    def init_model(config: NfnetConfig):
        model = RoiAttentionNFnet(
            num_classes = config.num_classes,
            variant = config.nf_variant, 
            stochdepth_rate=config.stochdepth_rate, 
            alpha=config.alpha,
            se_ratio=config.se_ratio,
            activation=config.activation,
        )
        model.to(config.device)
        return model
    
    @classmethod
    def start_new_training_from_pretrained(cls, pretrained_path: str, config: NfnetConfig):

        checkpoint = torch.load(pretrained_path)

        model = cls.init_model(config)
        model.load_pretrained_nfnet_weights(checkpoint["model_state_dict"])
        

        return super().start_new_training(model, config)
    
    def _train_epoch(self, train_dataset: SpotDataset):
        self.model.train()
        running_loss = 0.0
        correct_labels = 0
        # is_nan = False
        step = 0
        pbar = tqdm(train_dataset.dataloader, total=self.config.nf_iters_per_epoch)
        for inputs, rois in pbar:
            rois: dict
            step += 1

            inputs: Tensor = inputs.half().to(self.config.device)\
                if self.config["use_fp16"] else inputs.to(self.config.device)
            
            for key, value in rois.items():
                value: Tensor
                rois[key] = value.to(self.config.device)

            targets: Tensor = (rois["size"] > 0).to(torch.long)

            self.optimizer.zero_grad()

            with amp.autocast(enabled=self.config["amp"]):
                output = self.model(inputs, rois)
            loss: Tensor = self.criterion(output, targets)
            
            # Gradient scaling
            # https://www.youtube.com/watch?v=OqCrNkjN_PM
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct = (predicted == targets).sum().item()
            white_correct = ((predicted == 1) & (targets == 1)).sum().item()
            black_correct = ((predicted == 0) & (targets == 0)).sum().item()
            correct_labels += correct
            pbar.set_description(
                f"L: {loss.item():.2e}| #C: {correct:2d}| WC: {white_correct}| BC: {black_correct}")
            if step >= pbar.total:
                break

        running_loss = running_loss / step
        train_acc = correct_labels / (step * train_dataset.config.batch_size["train"])
        return Criteria(
            Loss(running_loss),
            Accuracy(train_acc)
        )
    
    @torch.no_grad()
    def _eval_epoch(self, eval_dataset: SpotDataset):
        self.model.eval()

        correct_labels = 0
        eval_loss = 0.0
        fp_val = 0.0
        step = 0
        metrics = Metrics(0, beta=self.config.f_score_beta)

        for inputs, rois in tqdm(eval_dataset.dataloader, disable=True):
            rois: dict
            step += 1
            inputs: Tensor = inputs.to(self.config.device)

            for key, value in rois.items():
                value: Tensor
                rois[key] = value.to(self.config.device)

            num_white: Tensor = rois["size"]
            targets: Tensor = (num_white > 0).to(torch.long)

            output: Tensor = self.model(inputs, rois).type(torch.float32)
            output = output.softmax(dim=1)

            metrics.update(output, targets)

            loss: Tensor = self.criterion.forward(output, targets)
            eval_loss += loss.item()
            _, predicted = torch.max(output, 1)
            print(predicted)
            print(num_white)
            fp_val += (output[:, 1] * num_white).sum().item()
            correct_labels += (predicted == targets).sum().item()

        eval_loss = eval_loss / step
        eval_acc = correct_labels / (step * eval_dataset.config.batch_size["val"])
        fp_val /= (step * eval_dataset.config.batch_size["val"])
        print(metrics.fp)
        print(metrics.fn)
        return Criteria(
            Loss(eval_loss),
            Accuracy(eval_acc),
            Precision(metrics.precision),
            Recall(metrics.recall),
            FScore(metrics.f_beta),
            FPVal(fp_val),
        )

    @torch.inference_mode()
    def validate_roi(
        self, img_path: str, rois_: List[ROI], record_exp_val: bool = False
    ) -> Tuple[List[bool], Union[List[float], None]]:
        """validate givien roi in rois

        Args:
            img_path (str): path to an image
            rois (List[ROI]): rois
        
        Returns:
            to_rm_spots (List[bool]): indices of rois which failed validation.
                (i.e. spots to be remove)
        """
        self.model.eval()
        ds = SingleImageRoiSpotDataset(img_path, rois_, self.config)
        num_roi = len(rois_)
        to_rm_spots: Tensor = torch.empty([num_roi], dtype=bool)
        if record_exp_val:
            expected_val: Tensor = torch.empty([num_roi], dtype=torch.float32) ## debugging
        else:
            expected_val = None

        for img, rois, indices in ds.dataloader:
            img: Tensor
            rois: dict
            img = img.to(self.config.device)
            for key, value in rois.items():
                value: Tensor
                rois[key] = value.to(self.config.device)

            preds: Tensor = self.model(img, rois)
            white_confidence = preds.softmax(dim=1)[:, 1].cpu().to(torch.float32)

            passed_spots = white_confidence > self.config.valid_spot_confidence_upper_bound
            failed_spots = white_confidence < self.config.valid_spot_confidence_lower_bound
            expected_fp = rois["size"].cpu().to(torch.float32) * white_confidence
            spot_to_rm = expected_fp < self.config.valid_spot_exp_val_threshold
            to_rm_spots[indices] = spot_to_rm
            to_rm_spots[indices[failed_spots]] = True
            to_rm_spots[indices[passed_spots]] = False
            if record_exp_val:
                expected_val[indices] = expected_fp
            # E(False Positive) = roi_size * P(spot_category = white)
            # if E(FP) < threshold, then remove this spot
            # else keep it
        return to_rm_spots.tolist(), expected_val.tolist() if record_exp_val else None
        
    