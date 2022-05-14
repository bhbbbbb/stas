from model_utils.base.criteria import Criteria
from model_utils.base.criteria import Loss as BaseLoss
from model_utils.base.criteria import Accuarcy as BaseAcc




Loss = Criteria.register_criterion(
    primary=False,
    plot=True,
)(BaseLoss)

# @Criteria.register_criterion()
# class Dice(BaseAcc):
#     short_name: str = 'dice'
#     full_name: str = 'Dice'
#     plot: bool = True

#     default_lower_limit_for_plot: float = 0.0
#     default_upper_limit_for_plot: float = 1.0

#     primary: bool = True

# @Criteria.register_criterion()
# class MIou(BaseAcc):
#     short_name: str = 'miou'
#     full_name: str = 'mIOU'
#     plot: bool = True

#     default_lower_limit_for_plot: float = 0.0
#     default_upper_limit_for_plot: float = 1.0

#     primary: bool = False

@Criteria.register_criterion()
class BackgroundIou(BaseAcc):
    short_name: str = 'bgiou'
    full_name: str = 'Iou_Bg'
    plot: bool = False

    default_lower_limit_for_plot: float = 0.0
    default_upper_limit_for_plot: float = 1.0

    primary: bool = False

@Criteria.register_criterion()
class TargetIou(BaseAcc):
    short_name: str = 'tgtiou'
    full_name: str = 'Iou_Target'
    plot: bool = True

    default_lower_limit_for_plot: float = 0.0
    default_upper_limit_for_plot: float = 1.0

    primary: bool = False



# @Criteria.register_criterion()
# class MAccuracy(BaseAcc):
#     short_name: str = 'macc'
#     full_name: str = 'mAcc'
#     plot: bool = True

#     default_lower_limit_for_plot: float = 0.0
#     default_upper_limit_for_plot: float = 1.0

#     primary: bool = False

@Criteria.register_criterion()
class BackgroundAccuracy(BaseAcc):
    short_name: str = 'bgacc'
    full_name: str = 'Acc_Bg'
    plot: bool = False

    default_lower_limit_for_plot: float = 0.0
    default_upper_limit_for_plot: float = 1.0

    primary: bool = False

@Criteria.register_criterion()
class TargetAccuracy(BaseAcc):
    short_name: str = 'tgtacc'
    full_name: str = 'Acc_Target'
    plot: bool = False

    default_lower_limit_for_plot: float = 0.0
    default_upper_limit_for_plot: float = 1.0

    primary: bool = False

# @Criteria.register_criterion()
# class MF1(BaseAcc):
#     short_name: str = 'mf1'
#     full_name: str = 'mF1'
#     plot: bool = True

#     default_lower_limit_for_plot: float = 0.0
#     default_upper_limit_for_plot: float = 1.0

#     primary: bool = False

@Criteria.register_criterion()
class BackgroundF1(BaseAcc):
    short_name: str = 'bgf1'
    full_name: str = 'F1_Bg'
    plot: bool = False

    default_lower_limit_for_plot: float = 0.0
    default_upper_limit_for_plot: float = 1.0

    primary: bool = False

@Criteria.register_criterion()
class TargetF1(BaseAcc):
    short_name: str = 'tgtf1'
    full_name: str = 'F1_Target'
    plot: bool = True

    default_lower_limit_for_plot: float = 0.0
    default_upper_limit_for_plot: float = 1.0

    primary: bool = True
