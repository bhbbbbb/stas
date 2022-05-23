from torch import Tensor

class Metrics:
    def __init__(self, true_label: int) -> None:
        self.true_label = true_label
        self.false_label = int(not true_label)
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

    def compute(self):
        """Compute precision recall and f1

        Returns:
            prescision, recall, f1
        """
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        # f1 = 2 * (precision * recall) / (precision + recall)
        f1 = self.tp / (self.tp + 0.5 * (self.fp + self.fn))
        return precision, recall, f1

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self):
        return self.tp / (self.tp + 0.5 * (self.fp + self.fn))

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
