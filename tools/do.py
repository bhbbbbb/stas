# pylint: disable=all
import os

import torch

from semseg.models.segformer import SegFormer
from model_utils import Criteria

from stas.config import Config as StasConfig
from stas.stas_dataset import StasDataset # , get_labels_ratio
from stas.stas_model_utils import StasModelUtils


from spot_validate.model import MyNfnet
from spot_validate.nfnet_model_utils import NfnetModelUtils, NfnetConfig
from spot_validate.dataset import SpotDataset, DatasetConfig as SpotDatasetConfig

try:
    from .inference import inference, inference_by_valid
except ImportError:
    from inference import inference, inference_by_valid




class Config(StasConfig, SpotDatasetConfig, NfnetConfig):
    DATASET_ROOT: str = os.path.join(__file__, '..', '..', '..', 'SEG_Train_Datasets')
    IMGS_ROOT: str = os.path.join(DATASET_ROOT, 'Train_Images')
    MASK_ROOT: str = os.path.join(DATASET_ROOT, 'Train_Masks')
    TRAIN_SPLIT: str = os.path.join(DATASET_ROOT, 'split_train.json')
    VALID_SPLIT: str = os.path.join(DATASET_ROOT, 'split_valid.json')
    SPOT_CSV: str = os.path.join(DATASET_ROOT, 'spot.csv')
    ROIS_ROOT: str = os.path.join(__file__, '..', '..', 'rois')

    log_dir = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\src\\log'
    pretrained = os.path.join(__file__, '..', '..', 'pretrained_models', 'mit_b0.pth')
    STD_SCLAER_DIR: str = os.path.join(log_dir, 'std_scaler')
    random_crop: bool = True

def main():
    """for debuging, not necessary"""
    torch.set_printoptions(edgeitems=10, linewidth=300)

    config = Config()
    config.batch_size['val'] = 16
    config.batch_size['train'] = 4
    config.epochs_per_checkpoint = 0
    config.epochs_per_eval = 1
    config.save_best = True
    config.nf_variant = 'F1'
    # config.train_size = config.inf_size
    # config.val_size = config.train_size

    # config.learning_rate *= 0.5
    config.learning_rate *= 0.07
    config.positive_threshold = 0.01
    config.small_spot_threshold = 200
    ### Test est level
    # config.batch_size['val'] = 1
    # config.show_progress_bar = False
    ###

    """check implementation and all the registed checking hooks
        and freeze(cannot reassign attr. of config anymore.)"""
    config.check_and_freeze()

    """display configurations to console"""
    config.display()

    # train_set = StasDataset(config, 'train')
    # valid_set = StasDataset(config, 'val')
    # train_set = SpotEstDataset(config, 'train')
    # valid_set = SpotEstDataset(config, 'val')
    train_set = SpotDataset(config, 'train')
    valid_set = SpotDataset(config, 'val')

    # model = SegFormer(config.backbone, config.num_classes)
    # model = SpotEst(num_classes=2)

    """start new training"""
    # utils = StasModelUtils.start_new_training(model, config)
    # pretrained_path = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\src\\log\\20220516T00-37-26\\20220516T18-00-00_epoch_157'
    # utils = SpotEstModelUtils.start_new_training_(model, config, pretrained_path)
    # NFNET_PRTRAINED = 'D:\\Documents\\PROgram\\ML\\kaggle\\crop-clef\\src\\pretrained_weights\\F1_haiku.npz'
    # NFNET_PRTRAINED = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\nfnet_pretrained\\F0_haiku.npz'
    # utils = NfnetModelUtils.start_new_training_from_pretrained(NFNET_PRTRAINED, config)


    """load from last checkpoint"""
    # utils = StasModelUtils.load_last_checkpoint(model, config)
    # utils = SpotEstModelUtils.load_last_checkpoint(model, config)
    model = NfnetModelUtils.init_model(config)
    utils = NfnetModelUtils.load_last_checkpoint(model, config)

    """or load from particular checkpoint"""
    # root = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\src\\log\\20220522T15-58-52_F0'
    # path = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\src\\log\\20220516T00-37-26\\20220517T13-46-48_epoch_160'
    # utils = StasModelUtils.load_checkpoint(model, path, config)
    # utils = SpotEstModelUtils.load_checkpoint(model, path, config)
    # model = NfnetModelUtils.init_model(config)
    # utils = NfnetModelUtils.load_last_checkpoint_from_dir(model, root, config)

    # utils.model.freeze_backbone()
    epochs = 50
    # utils._eval_epoch(valid_set).display()
    utils.train(epochs, train_set, valid_set)
    # con = Criteria.get_plot_configs_from_registered_criterion()
    # con[''].default_upper_limit_for_plot = 1e5
    # # con['spot_loss'].default_lower_limit_for_plot = 0
    # utils._eval_epoch(valid_set).display()
    utils.plot_history()
    # inf_set = SpotEstDataset(config, 'inference')
    # utils.inference(inf_set, utils.root)

    # do inference
    # utils.generate_rois()
    # PUBLIC_SET_ROOT = 'D:\\Documents\\PROgram\\ML\\kaggle\\stas-seg\\Public_Image'
    # inference(config, utils, test_dir=PUBLIC_SET_ROOT, num_output=-1)
    # inference_by_valid(config, utils, 20)

    print('\a') # finish alert
    return

    

if __name__ == '__main__':
    main()
