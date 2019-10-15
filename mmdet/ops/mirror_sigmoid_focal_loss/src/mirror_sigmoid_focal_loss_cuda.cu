#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <cfloat>


#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void MirrorSigmoidFocalLossForward(const int nthreads,
                                              const scalar_t *logits,
                                              const int64_t *targets,
                                              const int num_classes,
                                              const float gamma_1,
                                              const float gamma_2,
                                              const float alpha,
                                              const float thresh,
                                              const float beta,
                                              const int num, 
                                              scalar_t *losses) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        int n = i / num_classes;
        int d = i % num_classes;
        int t = targets[n];

        scalar_t c1 = (t == (d + 1));               //positive flag
        scalar_t c2 = (t >= 0 & t != (d + 1));      //negative flag

        scalar_t zp = alpha;
        scalar_t zn = (1.0 - alpha);

        // p = 1. / 1. + expf(-x); p = sigmoid(x)
        scalar_t p = 1.0 / (1.0 + expf(-logits[i]));
        scalar_t thr = (1.0 - p) >= thresh; // negative pt >= thresh, consider easy negative, otherwise hard negative.

        // (1-p)**gamma * log(p)
        scalar_t pos_term = powf((1.0-p), gamma_1) * logf(max(p, FLT_MIN));
        // p**gamma * log(1-p)
        scalar_t neg_term_easy = powf(p, gamma_1) * (-1.0 * logits[i] * (logits[i] >= 0) - logf(1.0 + expf(logits[i] - 2.*logits[i] * (logits[i] >= 0))));
        scalar_t neg_term_hard = powf((1.0 - p), gamma_2) * logf(max(p, FLT_MIN));

        losses[i] = 0.0;
        losses[i] += -c1 * pos_term * zp;
        losses[i] += -c2 * neg_term_easy * zn * thr;
        losses[i] += -c2 * neg_term_hard * zn * (1.0 - thr) * beta;
    }
}

template <typename scalar_t>
__global__ void MirrorSigmoidFocalLossBackward(const int nthreads,
                                               const scalar_t *logits,
                                               const int64_t *targets,
                                               const scalar_t *d_losses,
                                               const int num_classes,
                                               const float gamma_1,
                                               const float gamma_2,
                                               const float alpha,
                                               const float thresh,
                                               const float beta,
                                               const int num, 
                                               scalar_t *d_logits) {    
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        int n = i / num_classes;
        int d = i % num_classes;
        int t = targets[n];

        scalar_t c1 = (t == (d + 1));               //positive flag
        scalar_t c2 = (t >= 0 & t != (d + 1));      //negative flag

        scalar_t zp = alpha;
        scalar_t zn = (1.0 - alpha);

        // p = 1. / 1. + expf(-x); p = sigmoid(x)
        scalar_t p = 1.0 / (1.0 + expf(-logits[i]));
        scalar_t thr = (1.0 - p) >= thresh;

        scalar_t pos_term = powf((1.0 - p), gamma_1) * (1.0 - p - (p * gamma_1 * logf(max(p, FLT_MIN))));
        scalar_t neg_term_easy = 
            powf(p, gamma_1) * ((-1. * logits[i] * (logits[i] >= 0) - logf(1. + expf(logits[i] - 2. * logits[i] * (logits[i] >= 0)))) * (1. - p) * gamma_1 - p);
        scalar_t neg_term_hard = powf((1.0 - p), gamma_2) * (1.0 - p - (p * gamma_2 * logf(max(p, FLT_MIN))));

        d_logits[i] = 0.0;
        d_logits[i] += -c1 * pos_term * zp;
        d_logits[i] += -c2 * neg_term_easy * zn * thr;
        d_logits[i] += -c2 * neg_term_hard * zn * (1.0 - thr) * beta;
        d_logits[i] = d_logits[i] * d_losses[i];
    }
}

at::Tensor MirrorSigmoidFocalLoss_forward_cuda(const at::Tensor &logits,
                                                        const at::Tensor &targets,
                                                        const int num_classes,
                                                        const float gamma_1, const float gamma_2,
                                                        const float alpha, const float thresh,
                                                        const float beta) {
    AT_ASSERTM(logits.type().is_cuda(), "logits must be a CUDA tensor");
    AT_ASSERTM(targets.type().is_cuda(), "targets must be a CUDA tensor");
    AT_ASSERTM(logits.dim() == 2, "logits should be N x Num_classes");

    const int num_samples = logits.size(0);
    auto losses = at::empty({num_samples, logits.size(1)}, logits.options());
    auto losses_size = num_samples * logits.size(1);

    dim3 grid(std::min(THCCeilDiv((int64_t)losses_size, (int64_t)512), (int64_t)4096));
    dim3 block(512);

    if (losses.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return losses;
    }
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        logits.scalar_type(), "MirrorSigmoidFocalLoss_forward", [&] {
                MirrorSigmoidFocalLossForward<scalar_t><<<grid, block>>>(
                losses_size, logits.contiguous().data<scalar_t>(),
                targets.contiguous().data<int64_t>(), num_classes, gamma_1, 
                gamma_2, alpha, thresh, beta, num_samples, losses.data<scalar_t>());
        });
    THCudaCheck(cudaGetLastError());
    return losses;
}

at::Tensor MirrorSigmoidFocalLoss_backward_cuda(const at::Tensor &logits,
                                                const at::Tensor &targets,
                                                const at::Tensor &d_losses,
                                                const int num_classes,
                                                const float gamma_1, const float gamma_2,
                                                const float alpha, const float thresh,
                                                const float beta) {

    AT_ASSERTM(logits.type().is_cuda(), "logits must be a CUDA tensor");
    AT_ASSERTM(targets.type().is_cuda(), "targets must be a CUDA tensor");
    AT_ASSERTM(d_losses.type().is_cuda(), "logits should be NxClass");

    AT_ASSERTM(logits.dim() == 2, "logits should be N x num_classes");

    const int num_samples = logits.size(0);
    AT_ASSERTM(logits.size(1) == num_classes, "logits.size(1) should equal to num_classes");
    
    auto d_logits = at::zeros({num_samples, num_classes}, logits.options());
    auto d_logits_size = num_samples * logits.size(1);
    
    dim3 grid(std::min(THCCeilDiv((int64_t)d_logits_size, (int64_t)512), (int64_t)4096));
    dim3 block(512);

    if (d_logits.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return d_logits;
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        logits.scalar_type(), "MirrorSigmoidFocalLoss_backward", [&] {
            MirrorSigmoidFocalLossBackward<scalar_t><<<grid, block>>>(
                d_logits_size, logits.contiguous().data<scalar_t>(),
                targets.contiguous().data<int64_t>(),
                d_losses.contiguous().data<scalar_t>(),
                num_classes,
                gamma_1, gamma_2, alpha, thresh, beta, num_samples, d_logits.data<scalar_t>()
            );
        }
    );
    THCudaCheck(cudaGetLastError());
    return d_logits;
}

