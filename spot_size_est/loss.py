from torch import Tensor, nn

class L1L2Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='sum')
        self.l2 = nn.MSELoss(reduction='sum')
        return
    
    def forward(self, x: Tensor, y: Tensor):
        mask = x < y
        x_left = x * mask
        y_left = y * mask
        left_loss = self.l1(x_left, y_left)
        mask = x > y
        x_right = x * mask
        y_right = y * mask
        right_loss = self.l2(x_right, y_right)
        return (left_loss + right_loss) / x.shape[0]
