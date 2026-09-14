#pragma once
#include <type_define.h>
#include <cuda_tool/cuda_tool.h>

namespace uipc::backend::cuda
{
#ifdef __CUDACC__
/**
 * @brief Device-side switch read by every stencil Hessian projection: 1 (the
 * default) clamps negative eigenvalues to zero, 0 leaves the Hessian as
 * derived. The forward Newton solve always projects; only the adjoint
 * export's re-assembly at the accepted state turns projection off.
 * Defined in linear_system/linear_system_adjoint.cu.
 */
extern __device__ int uipc_hessian_projection_enabled;
#endif

/// Set the device-side projection switch; synchronises the device first.
void set_hessian_projection(bool enabled);
/// The host's record of the switch (true until set otherwise).
bool hessian_projection_enabled();

template <int N>
UIPC_GENERIC void make_spd(Matrix<Float, N, N>& H)
{
#if defined(__CUDA_ARCH__)
    if(!uipc_hessian_projection_enabled)
        return;
#endif
    Vector<Float, N>    eigen_values;
    Matrix<Float, N, N> eigen_vectors;
    cuda_tool::eigen::template evd<Float, N>(H, eigen_values, eigen_vectors);
    for(int i = 0; i < N; ++i)
    {
        auto& v = eigen_values(i);
        v       = v < 0.0 ? 0.0 : v;
    }
    H = eigen_vectors * eigen_values.asDiagonal() * eigen_vectors.transpose();
}
}  // namespace uipc::backend::cuda