// modify from
// https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/csrc/SigmoidFocalLoss.h
#include <torch/extension.h>

at::Tensor SigmoidFocalLossMirror_forward_cuda(const at::Tensor &logits,
                                                const at::Tensor &targets,
                                                const int num_classes,
                                                const float gamma, const float alpha,
                                                const float gamma2, const float thresh);

at::Tensor SigmoidFocalLossMirror_backward_cuda(const at::Tensor &logits,
                                                const at::Tensor &targets,
                                                const at::Tensor &d_losses,
                                                const int num_classes,
                                                const float gamma, const float alpha,
                                                const float gamma2, const float thresh);

// Interface for Python
at::Tensor SigmoidFocalLossMirror_forward(const at::Tensor &logits,
                                            const at::Tensor &targets,
                                            const int num_classes, const float gamma,
                                            const float alpha,
                                            const float gamma2, const float thresh) {
  if (logits.type().is_cuda()) {
    return SigmoidFocalLossMirror_forward_cuda(logits, targets, num_classes, gamma, alpha, gamma2, thresh);
  }
}

at::Tensor SigmoidFocalLossMirror_backward(const at::Tensor &logits,
                                            const at::Tensor &targets,
                                            const at::Tensor &d_losses,
                                            const int num_classes, const float gamma,
                                            const float alpha,
                                            const float gamma2,
                                            const float thresh) {
  if (logits.type().is_cuda()) {
    return SigmoidFocalLossMirror_backward_cuda(logits, targets, d_losses, num_classes, gamma, alpha, gamma2, thresh);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &SigmoidFocalLossMirror_forward,
        "SigmoidFocalLossMirror forward (CUDA)");
  m.def("backward", &SigmoidFocalLossMirror_backward,
        "SigmoidFocalLossMirror backward (CUDA)");
}
