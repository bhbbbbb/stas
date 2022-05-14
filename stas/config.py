import os
from typing import Tuple
from model_utils.config import ModelUtilsConfig
from .stas_dataset import DatasetConfig

class Config(ModelUtilsConfig, DatasetConfig):

    epochs_per_checkpoint: int = 1
    """num of epochs per checkpoints

        Example:
            1: stand for save model every epoch
            0: for not save until finish
    """

    log_dir: str = 'log'
    """dir for saving checkpoints and log files"""

    logging: bool = True
    """whether log to log.log. It's useful to turn this off when inference"""

    epochs_per_eval: int = 1
    """Number of epochs per evalution"""

    early_stopping: bool = False
    """whether enable early stopping"""

    early_stopping_threshold: int = 100
    """Threshold for early stopping mode. Only matter when EARLY_STOPPING is set to True."""

    save_best = False


    device = 'cuda:0'
    backbone: str = 'MiT-B0'
    pretrained: str = 'pretrained/MiT-B0'              # backbone model's weight 
    ignore_label: int = -1


    num_classes: int = 2
    img_size = (471, 858)
    DATASET_ROOT: str = '../SEG_Train_Datasets'
    IMGS_ROOT: str = os.path.join(DATASET_ROOT, 'Train_Images')
    MASK_ROOT: str = os.path.join(DATASET_ROOT, 'Train_Masks')
    TRAIN_SPLIT: str = os.path.join(DATASET_ROOT, 'split_train.json')
    VALID_SPLIT: str = os.path.join(DATASET_ROOT, 'split_valid.json')

    batch_size = {
        'train': 4,
        'val': 4,
    }

    num_workers: int = 4
    drop_last: bool = True
    pin_memory: bool = True

    # IMAGE_SIZE    : [512, 512]    # training image size in (h, w)
    # BATCH_SIZE    : 2               # batch size used to train
    # EPOCHS        : 100             # number of epochs to train

    # EVAL_INTERVAL : 20              # evaluation interval during training
    AMP: bool = False          # use AMP in training
    DDP: bool = False          # use DDP training

    loss_name: str = 'OhemCrossEntropy'
    cls_weights: list = [1/20, 19/20]


    optimizer_name: str = 'adamw'
    learning_rate: float = 0.001
    weight_decay: float = 0.01


    # SCHEDULER:
    scheduler_name: str = 'warmuppolylr'    # scheduler name
    power: float = 0.9             # scheduler power
    max_epoch: int = 200

    @property
    def max_iters(self):
        return self.max_epoch * 842 // self.batch_size['train']
    @property
    def warmup_iters(self):
        return 10 * 842 // self.batch_size['train']
    warmup_ratio: float = 0.1             # warmup ratio
    

    # EVAL:
    # MODEL_PATH    : 'checkpoints/pretrained/ddrnet/ddrnet_23slim_city.pth'     # trained model file path
    # IMAGE_SIZE    : [1024, 1024]                            # evaluation image size in (h, w)                       
    # MSF: 
    #     ENABLE      : false                                   # multi-scale and flip evaluation  
    #     FLIP        : true                                    # use flip in evaluation  
    #     SCALES      : [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]       # scales used in MSF evaluation                


    # TEST:
    # MODEL_PATH    : 'checkpoints/pretrained/ddrnet/ddrnet_23slim_city.pth'    # trained model file path
    # FILE          : 'assests/cityscapes'                    # filename or foldername 
    # IMAGE_SIZE    : [1024, 1024]                            # inference image size in (h, w)
    # OVERLAY       : true                                    # save the overlay result (image_alpha+label_alpha)