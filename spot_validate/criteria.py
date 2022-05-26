from model_utils import Criteria, Loss as BaseLoss, Accuarcy as BaseAcc

Loss = Criteria.register_criterion(
    short_name="nf_loss",
    plot=True,
    primary=False,
)(BaseLoss)

Accuracy = Criteria.register_criterion(
    short_name="nf_acc",
    plot=True,
    primary=False,
)(BaseAcc)

# call black but white
FPVal = Criteria.register_criterion(
    short_name="nf_expected_fp",
    full_name="E(FP)",
    plot=True,
    primary=False,
)(BaseLoss)

Precision = Criteria.register_criterion(
    short_name="precision",
    full_name="Precision",
    plot=False,
    primary=False,
)(BaseAcc)

Recall = Criteria.register_criterion(
    short_name="recall",
    full_name="Recall",
    plot=False,
    primary=False,
)(BaseAcc)

FScore = Criteria.register_criterion(
    short_name="nf_f_beta",
    full_name="F_beta",
    plot=True,
    primary=True,
)(BaseAcc)