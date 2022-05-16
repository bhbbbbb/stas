# pylint: disable=all
import os

import torch

from semseg.models.segformer import SegFormer

from stas.config import Config as StasConfig
from stas.stas_dataset import StasDataset # , get_labels_ratio
from stas.stas_model_utils import StasModelUtils

try:
    from .inference import inference, inference_by_valid
except ImportError:
    from inference import inference, inference_by_valid



class Config(StasConfig):
    DATASET_ROOT: str = os.path.join(__file__, '..', '..', '..', 'SEG_Train_Datasets')
    IMGS_ROOT: str = os.path.join(DATASET_ROOT, 'Train_Images')
    MASK_ROOT: str = os.path.join(DATASET_ROOT, 'Train_Masks')
    TRAIN_SPLIT: str = os.path.join(DATASET_ROOT, 'split_train.json')
    VALID_SPLIT: str = os.path.join(DATASET_ROOT, 'split_valid.json')

    log_dir = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\src\\log'
    pretrained = os.path.join('..', 'pretrained_models', 'mit_b0.pth')

def main():
    """for debuging, not necessary"""
    torch.set_printoptions(edgeitems=10, linewidth=300)

    config = Config()
    config.batch_size['val'] = 1
    config.epochs_per_checkpoint = 40
    config.epochs_per_eval = 1
    config.save_best = True

    """check implementation and all the registed checking hooks
        and freeze(cannot reassign attr. of config anymore.)"""
    config.check_and_freeze()

    """display configurations to console"""
    config.display()

    train_set = StasDataset(config, 'train')
    valid_set = StasDataset(config, 'val')

    model = SegFormer(config.backbone, config.num_classes)

    """start new training"""
    utils = StasModelUtils.start_new_training(model, config)

    """load from last checkpoint"""
    # utils = StasModelUtils.load_last_checkpoint(model, config)

    """or load from particular checkpoint"""
    # path = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\src\\log\\20220516T00-37-26\\20220516T08-57-33_epoch_118'
    # utils = StasModelUtils.load_checkpoint(model, path, config)

    epochs = 200
    utils.train(epochs, train_set, valid_set)
    utils.plot_history()

    # do inference
    # PUBLIC_SET_ROOT = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\Public_Image'
    # inference(config, utils, test_dir=PUBLIC_SET_ROOT, num_output=-1)
    inference_by_valid(config, utils, 5)

    print('\a') # finish alert
    return

    

if __name__ == '__main__':
    main()
