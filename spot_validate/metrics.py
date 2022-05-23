from torch import Tensor

class Metrics:
    def __init__(self, true_label: int, beta: float = 1) -> None:
        self.true_label = true_label
        self.false_label = int(not true_label)
        self.beta = beta
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        return

    def update(self, pred: Tensor, target: Tensor) -> None:
        # pred: b x c
        # target: b
        self.tp += (pred[:, self.true_label] * (target == self.true_label)).sum().item()
        self.tn += (pred[:, self.false_label] * (target == self.false_label)).sum().item()

        self.fp += (pred[:, self.true_label] * (target == self.false_label)).sum().item()
        self.fn += (pred[:, self.false_label] * (target == self.true_label)).sum().item()
        return

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn)

    @property
    def f_beta(self):
        # https://en.wikipedia.org/wiki/F-score
        tp_ = (1 + self.beta ** 2) * self.tp
        fn_ = (self.beta ** 2) * self.fn
        return tp_ / (tp_ + fn_ + self.fp)

    # def compute_iou(self) -> Tuple[Tensor, Tensor]:
    #     ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
    #     miou = ious[~ious.isnan()].mean().item()
    #     ious *= 100
    #     miou *= 100
    #     return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    # def compute_f1(self) -> Tuple[Tensor, Tensor]:
    #     f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
    #     mf1 = f1[~f1.isnan()].mean().item()
    #     f1 *= 100
    #     mf1 *= 100
    #     return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    # def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
    #     acc = self.hist.diag() / self.hist.sum(1)
    #     macc = acc[~acc.isnan()].mean().item()
    #     acc *= 100
    #     macc *= 100
    #     return acc.cpu().numpy().round(2).tolist(), round(macc, 2)
