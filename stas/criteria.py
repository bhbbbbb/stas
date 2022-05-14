from model_utils.base.criteria import Criteria
from model_utils.base.criteria import Loss as BaseLoss
from model_utils.base.criteria import Accuarcy as BaseAcc




Loss = Criteria.register_criterion(
    primary=False,
    plot=True,
)(BaseLoss)

@Criteria.register_criterion()
class MIou(BaseAcc):
    short_name: str = "miou"
    full_name: str = "mIOU"
    plot: bool = True

    default_lower_limit_for_plot: float = 0.0
    default_upper_limit_for_plot: float = 1.0

    primary: bool = True

@Criteria.register_criterion()
class MAccuracy(BaseAcc):
    short_name: str = "macc"
    full_name: str = "mAccuracy"
    plot: bool = False

    default_lower_limit_for_plot: float = 0.0
    default_upper_limit_for_plot: float = 1.0

    primary: bool = False

@Criteria.register_criterion()
class MF1(BaseAcc):
    short_name: str = "mf1"
    full_name: str = "mF-1"
    plot: bool = True

    default_lower_limit_for_plot: float = 0.0
    default_upper_limit_for_plot: float = 1.0

    primary: bool = False
