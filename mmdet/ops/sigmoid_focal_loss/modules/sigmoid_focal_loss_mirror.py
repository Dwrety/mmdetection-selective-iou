from torch import nn

from ..functions.sigmoid_focal_loss_mirror import sigmoid_focal_loss_mirror


class SigmoidFocalMirrorLoss(nn.Module):

    def __init__(self, gamma, alpha, gamma2, thresh):
        super(SigmoidFocalMirrorLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.gamma2 = gamma2
        self.thresh = thresh

    def forward(self, logits, targets):
        assert logits.is_cuda
        loss = sigmoid_focal_loss_mirror(logits, targets, self.gamma, self.alpha, self.gamma2, self.thresh)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ", gamma2=" + str(self.gamma2) 
        tmpstr += ", thresh=" + str(self.thresh)
        tmpstr += ")"
        return tmpstr


if __name__ == "__main__":

    loss = SigmoidFocalLoss(2.0, 0.25)
