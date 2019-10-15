import torch.nn.functional as F
from torch.autograd import Function, Variable
from torch.autograd.function import once_differentiable
from .. import sigmoid_focal_loss_mirror_cuda


class SigmoidFocalLossMirrorFunction(Function):

    @staticmethod
    def forward(ctx, input, target, gamma=2.0, alpha=0.25, gamma2=2.0, thresh=0.1, reduction='mean'):
        ctx.save_for_backward(input, target)
        num_classes = input.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha
        ctx.gamma2 = gamma2
        ctx.thresh = thresh

        loss = sigmoid_focal_loss_mirror_cuda.forward(input, target, num_classes,
                                               gamma, alpha, gamma2, thresh)
        # print("forward loss shape", loss.shape)
        reduction_enum = F._Reduction.get_enum(reduction)
        # none: 0, mean:1, sum: 2
        if reduction_enum == 0:
            return loss
        elif reduction_enum == 1:
            return loss.mean()
        elif reduction_enum == 2:
            return loss.sum()

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        input, target = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        gamma2 = ctx.gamma2
        thresh = ctx.thresh
        d_loss = d_loss.contiguous()
        # print("loss out shape", d_loss.shape)
        d_input = sigmoid_focal_loss_mirror_cuda.backward(input, target, d_loss,
                                                   num_classes, gamma, alpha, gamma2, thresh)
        return d_input, None, None, None, None, None, None


sigmoid_focal_loss_mirror = SigmoidFocalLossMirrorFunction.apply
