#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "grouping_cuda_kernel.h"

void grouping_forward_cuda(int b, int c, int n, int m, int nsample, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(points_tensor));

    const float *points = points_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *out = out_tensor.data_ptr<float>();
    grouping_forward_cuda_launcher(b, c, n, m, nsample, points, idx, out);
}

void grouping_backward_cuda(int b, int c, int n, int m, int nsample, at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_out_tensor));

    float *grad_points = grad_points_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    const float *grad_out = grad_out_tensor.data_ptr<float>();
    grouping_backward_cuda_launcher(b, c, n, m, nsample, grad_out, idx, grad_points);
}

void grouping_forward_cuda_fast(int b, int c, int n, int npoints, int nsample, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor)
{

    const at::cuda::OptionalCUDAGuard device_guard(device_of(points_tensor));

    const float *points = points_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *out = out_tensor.data_ptr<float>();
    grouping_forward_cuda_launcher_fast(b, c, n, npoints, nsample, points, idx, out);
}
