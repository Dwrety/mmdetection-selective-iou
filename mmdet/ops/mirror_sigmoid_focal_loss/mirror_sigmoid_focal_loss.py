import torch.nn as nn 
from torch.autograd import Function 
from torch.autograd.function import once_differentiable

from . import mirror_sigmoid_focal_loss_cuda

class MirrorSigmoidFocalLossFunction(Function):

    @staticmethod
    def forward(ctx, input, target, 
                gamma_1=2.0, gamma_2=2.0, alpha=0.25,
                thresh=0.05, beta=1.0):
        ctx.save_for_backward(input, target)
        num_classes = input.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma_1 = gamma_1
        ctx.gamma_2 = gamma_2
        ctx.alpha = alpha
        ctx.thresh = thresh
        ctx.beta = beta

        loss = mirror_sigmoid_focal_loss_cuda.forward(input, target, num_classes, gamma_1, gamma_2, alpha, thresh, beta)

        return loss


    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        input, target = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma_1 = ctx.gamma_1
        gamma_2 = ctx.gamma_2 
        alpha = ctx.alpha 
        thresh = ctx.thresh
        ctx.beta = beta = ctx.beta

        d_loss = d_loss.contiguous()
        d_input = mirror_sigmoid_focal_loss_cuda.backward(input, target, d_loss, num_classes, gamma_1, gamma_2, alpha, thresh, beta)

        return d_input, None, None, None, None, None, None, None

mirror_sigmoid_focal_loss = MirrorSigmoidFocalLossFunction.apply


if __name__ == "__main__":
    test_inputs = torch.rand(16, 20)
    targets = torch.ones(16).long()

    print(test_inputs, targets)