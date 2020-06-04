import torch.nn as nn
import torch.nn.functional as F 

from mmdet.ops import mirror_sigmoid_focal_loss as _mirror_sigmoid_focal_loss

from ..registry import LOSSES
from .utils import weight_reduce_loss


def mirror_sigmoid_focal_loss(pred,
                              target,
                              weight=None,
                              gamma_1=2.0,
                              gamma_2=2.0,
                              alpha=0.25,
                              thresh=0.10,
                              beta=1.0,
                              reduction='mean',
                              avg_factor=None):
    loss = _mirror_sigmoid_focal_loss(pred, target, gamma_1, gamma_2, alpha, thresh, beta)
    if weight is not None:
        weight = weight.contiguous().view(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

@LOSSES.register_module
class MirrorFocalLoss(nn.Module):
    """This is the implementation of Background Recalibration FocalLoss.
       This function needs to work with special anchor box thresholding.
    """
    def __init__(self, use_sigmoid=True,
                 gamma_1=2.0,
                 gamma_2=2.0,
                 alpha=0.25,
                 thresh=0.10,
                 beta=0.0,
                 reduction='mean',
                 loss_weight=1.):

        super(MirrorFocalLoss, self).__init__()
        assert use_sigmoid is True
        self.use_sigmoid = use_sigmoid
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.alpha = alpha
        self.thresh = thresh
        self.beta = beta 
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        if self.use_sigmoid:
            loss_cls = self.loss_weight * mirror_sigmoid_focal_loss(pred,
                                                                    target,
                                                                    weight,
                                                                    gamma_1=self.gamma_1,
                                                                    gamma_2=self.gamma_2,
                                                                    alpha=self.alpha,
                                                                    thresh=self.thresh,
                                                                    beta=self.beta,
                                                                    reduction=reduction,
                                                                    avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls













