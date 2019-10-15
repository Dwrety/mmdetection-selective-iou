#include <torch/extension.h>

at::Tensor MirrorSigmoidFocalLoss_forward_cuda(const at::Tensor &logits,
                                               const at::Tensor &targets,
                                               const int num_classes,
                                               const float gamma_1,
                                               const float gamma_2,
                                               const float alpha,
                                               const float thresh,
                                               const float beta);

at::Tensor MirrorSigmoidFocalLoss_backward_cuda(const at::Tensor &logits,
                                              const at::Tensor &targets,
                                              const at::Tensor &d_losses,
                                              const int num_classes,
                                              const float gamma_1,
                                              const float gamma_2,
                                              const float alpha,
                                              const float thresh,
                                              const float beta);

at::Tensor MirrorSigmoidFocalLoss_forward(const at::Tensor &logits,
                                        const at::Tensor &targets,
                                        const int num_classes,
                                        const float gamma_1,
                                        const float gamma_2,
                                        const float alpha,
                                        const float thresh,
                                        const float beta) {
/*
    :gamma_1: original gamma value in FocalLoss;
    :gamma_2: mirrored part gamma, used to tune gradient in the hard negative region;
    :thresh: float number between 0, 1, indicating hard threshold;
    :beta: constant variable that applied only to the hard negative region;
*/

    if (logits.type().is_cuda()) { 
        return MirrorSigmoidFocalLoss_forward_cuda(logits, targets, num_classes, 
                                                    gamma_1, gamma_2, alpha, thresh, beta);
    }
    AT_ERROR("Only support GPU SigmoidFocalLoss");                                   
}
    
at::Tensor MirrorSigmoidFocalLoss_backward(const at::Tensor &logits,
                                            const at::Tensor &targets,
                                            const at::Tensor &d_losses,
                                            const int num_classes,
                                            const float gamma_1,
                                            const float gamma_2,
                                            const float alpha,
                                            const float thresh,
                                            const float beta){
    if (logits.type().is_cuda()) { 
        return MirrorSigmoidFocalLoss_backward_cuda(logits, targets, d_losses, num_classes, 
                                                    gamma_1, gamma_2, alpha, thresh, beta);
    }
    AT_ERROR("Only support GPU SigmoidFocalLoss");                                   
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &MirrorSigmoidFocalLoss_forward, 
          "GPU MirrorSigmoidFocalLoss forward method (CUDA)");
    m.def("backward", &MirrorSigmoidFocalLoss_backward, 
          "GPU MirrorSigmoidFocalLoss forward method (CUDA)");
}