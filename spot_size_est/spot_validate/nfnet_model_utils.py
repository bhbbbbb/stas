import torch
from torch import Tensor
from torch import nn
from torch.cuda import amp

from tqdm import tqdm

from model_utils import BaseModelUtils, Criteria, Loss as BaseLoss, Accuarcy as BaseAcc
from nfnets import SGD_AGC, pretrained_nfnet, NFNet # pylint: disable=import-error

from .dataset import SpotDataset
from .model import MyNfnet
from .config import NfnetConfig

Loss = Criteria.register_criterion(
    short_name="nf_loss",
    plot=True,
    primary=False
)(BaseLoss)

Accuracy = Criteria.register_criterion(
    short_name="nf_acc",
    plot=True,
    primary=True
)(BaseAcc)


class NfnetModelUtils(BaseModelUtils):

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
        self.criterion = nn.CrossEntropyLoss()
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
    def _get_optimizer(model: NFNet, config: NfnetConfig):

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
        model = MyNfnet(
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

        pretrained_model = pretrained_nfnet(pretrained_path)
        model = cls.init_model(config)
        model_state = pretrained_model.state_dict()
        model_state = MyNfnet.fix_output_layer(model_state, config.num_classes)
        
        model.load_state_dict(model_state)

        return super().start_new_training(model, config)
    
    def _train_epoch(self, train_dataset: SpotDataset):
        self.model.train()
        running_loss = 0.0
        correct_labels = 0
        # is_nan = False
        step = 0
        pbar = tqdm(train_dataset.dataloader)
        for inputs, targets in pbar:
            step += 1

            inputs: Tensor = inputs.half().to(self.config.device)\
                if self.config["use_fp16"] else inputs.to(self.config.device)
            
            targets: Tensor = targets.to(self.config.device)

            self.optimizer.zero_grad()

            with amp.autocast(enabled=self.config["amp"]):
                output = self.model(inputs)
            loss: Tensor = self.criterion(output, targets)
            
            # Gradient scaling
            # https://www.youtube.com/watch?v=OqCrNkjN_PM
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct = (predicted == targets).sum().item()
            correct_labels += correct
            pbar.set_description(f"L: {loss.item():.2e}| #C: {correct:2d}")

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
        step = 0
        for inputs, targets in tqdm(eval_dataset.dataloader):
            step += 1
            inputs: Tensor = inputs.to(self.config.device)
            targets: Tensor = targets.to(self.config.device)

            output = self.model(inputs).type(torch.float32)

            loss: Tensor = self.criterion.forward(output, targets)
            eval_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_labels += (predicted == targets).sum().item()

        eval_loss = eval_loss / step
        eval_acc = correct_labels / (step * eval_dataset.config.batch_size["val"])
        return Criteria(
            Loss(eval_loss),
            Accuracy(eval_acc)
        )

    # def inference(self, dataset: SpotDataset, categories: list = None, confidence: bool = True):
    #     """inference for the given test dataset

    #     Args:
    #         confidence (boolean): whether output the `confidence` column. Default to True.
        
    #     Returns:
    #         df (pd.DataFrame): {"label": [...], "confidence"?: [...]}
    #     """

    #     categories = categories if categories is not None else list(range(self.config.num_classes))
        
    #     def mapping(x):
    #         return categories[x]

    #     label_col = np.empty(len(dataset), dtype=type(categories[0]))
    #     if confidence:
    #         confidence_col = np.empty(len(dataset), dtype=float)
    #         data = {"label": label_col, "confidence": confidence_col}
        
    #     else:
    #         data = {"label": label_col}
        
    #     df = pd.DataFrame(data)
        
    #     with torch.inference_mode():
    #         for data, indexes in tqdm(dataset.dataloader):
    #             data: Tensor
    #             indexes: Tensor
    #             data = data.to(self.config.device)
    #             output: Tensor = self.model(data)

    #             output = F.softmax(output, dim=1)

    #             confidences, indices = output.max(dim=1)

    #             labels = list(map(mapping, indices.tolist()))
                
    #             indexes = indexes.tolist()

    #             df.loc[indexes, "label"] = labels
    #             if confidence:
    #                 df.loc[indexes, "confidence"] = confidences.tolist()
        
    #     return df
