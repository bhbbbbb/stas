from model_utils.config import ModelUtilsConfig

class NfnetConfig(ModelUtilsConfig):
    amp = False        # Enable automatic mixed precision

    # Model
    nf_variant = 'F0'         # F0 - F7
    num_classes = 2     # Number of classes
    activation = 'gelu'    # or 'relu'
    stochdepth_rate = 0.25 # 0-1, the probability that a layer is dropped during one step
    alpha = 0.2            # Scaling factor at the end of each block
    se_ratio = 0.5         # Squeeze-Excite expansion ratio
    use_fp16 = False       # Use 16bit floats, which lowers memory footprint. This currently sets
                        # the complete model to FP16 (will be changed to match FP16 ops from paper)

    # Training
    # batch_size = 64        # Batch size
    # epochs = 360           # Number of epochs
    # overfit = False        # Train on one batch size only

    # learning_rate = 0.1    # Learning rate
    # scale_lr = True        # Scale learning rate with batch size. lr = lr*batch_size/256
    momentum = 0.9         # Contribution of earlier gradient to gradient update
    weight_decay = 0.00002 # Factor with which weights are added to gradient
    nesterov = True        # Enable nesterov correction

    do_clip = True         # Enable adaptive gradient clipping
    clipping = 0.1         # Adaptive gradient clipping parameter

    epochs_per_checkpoint: int = 0
    
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

    show_progress_bar: bool = True
    save_best = False


    device = 'cuda:0'

    f_score_beta: float = 0.5
    # 0.5: importance of precision > recall
    # https://en.wikipedia.org/wiki/F-score

    valid_spot_confidence_lower_bound: float = 0.25
    valid_spot_confidence_upper_bound: float = 0.75
    valid_spot_exp_val_threshold: float = 1000
    # see NfnetModelUtils.validate_roi
    batch_size: dict

    @property
    def nf_iters_per_epoch(self):
        return (2 ** 11) // self.batch_size['train']
    
    nf_cls_weight = (1, 2)


class CvtConfig(ModelUtilsConfig):
    epochs_per_checkpoint: int = 0
    
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

    show_progress_bar: bool = True
    save_best = False


    device = 'cuda:0'

    f_score_beta: float = 0.5
    # 0.5: importance of precision > recall
    # https://en.wikipedia.org/wiki/F-score

    valid_spot_confidence_lower_bound: float = 0.25
    valid_spot_confidence_upper_bound: float = 0.75
    valid_spot_exp_val_threshold: float = 1000
    # see NfnetModelUtils.validate_roi

    weight_decay: float = 0.05
    amp = False        # Enable automatic mixed precision

    # cvt_config = dict(
    #     s1_emb_dim = 64,
    #     s1_emb_kernel = 7,
    #     s1_emb_stride = 4,
    #     s1_proj_kernel = 3,
    #     s1_kv_proj_stride = 2,
    #     s1_heads = 1,
    #     s1_depth = 1,
    #     s1_mlp_mult = 4,
    #     s2_emb_dim = 192,
    #     s2_emb_kernel = 3,
    #     s2_emb_stride = 2,
    #     s2_proj_kernel = 3,
    #     s2_kv_proj_stride = 2,
    #     s2_heads = 3,
    #     s2_depth = 2,
    #     s2_mlp_mult = 4,
    #     s3_emb_dim = 384,
    #     s3_emb_kernel = 3,
    #     s3_emb_stride = 2,
    #     s3_proj_kernel = 3,
    #     s3_kv_proj_stride = 2,
    #     s3_heads = 6,
    #     s3_depth = 10,
    #     s3_mlp_mult = 4,
    #     dropout = 0.2
    # )
    # cvt_config = dict(
    #     s1_emb_dim = 192,
    #     s1_emb_kernel = 7,
    #     s1_emb_stride = 4,
    #     s1_proj_kernel = 3,
    #     s1_kv_proj_stride = 2,
    #     s1_heads = 3,
    #     s1_depth = 2,
    #     s1_mlp_mult = 4,
    #     s2_emb_dim = 768,
    #     s2_emb_kernel = 3,
    #     s2_emb_stride = 2,
    #     s2_proj_kernel = 3,
    #     s2_kv_proj_stride = 2,
    #     s2_heads = 12,
    #     s2_depth = 2,
    #     s2_mlp_mult = 4,
    #     s3_emb_dim = 1024,
    #     s3_emb_kernel = 3,
    #     s3_emb_stride = 2,
    #     s3_proj_kernel = 3,
    #     s3_kv_proj_stride = 2,
    #     s3_heads = 16,
    #     s3_depth = 20,
    #     s3_mlp_mult = 4,
    #     dropout = 0.
    # )
