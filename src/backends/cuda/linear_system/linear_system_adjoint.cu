#include <sim_system.h>
#include <linear_system/global_linear_system.h>
#include <uipc/diff_sim/linear_system_adjoint_feature.h>
#include <uipc/common/log.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace uipc::backend::cuda
{
/**
 * @brief Exposes the assembled system of the current frame and a solve
 * against it, for sensitivities computed outside the backend.
 *
 * `GlobalLinearSystem` keeps `bcoo_A` (the block-sparse Hessian after
 * symmetric compression, upper block triangle), `b` (the gradient) and `x`
 * (the last Newton step) from the frame's last iteration until the next
 * frame's first solve. Nothing here rebuilds the system; `do_solve` swaps a
 * caller's right-hand side in, runs the frame's solver on the frame's matrix
 * and preconditioner, and swaps the frame's own vectors back.
 */
class LinearSystemAdjointFeatureOverrider final : public diff_sim::LinearSystemAdjointFeatureOverrider
{
  public:
    explicit LinearSystemAdjointFeatureOverrider(GlobalLinearSystem& system)
        : m_system(system)
    {
    }

    SizeT get_dof_count() override { return m_system.m_impl.x.size(); }

    SizeT get_triplet_count() override
    {
        return m_system.m_impl.bcoo_A.triplet_count();
    }

    void do_export_system(span<IndexT> block_rows,
                          span<IndexT> block_cols,
                          span<Float>  block_values,
                          span<Float>  gradient) override
    {
        auto& impl = m_system.m_impl;
        UIPC_ASSERT_THROW(!impl.empty_system,
                          "LinearSystemAdjoint: no system has been assembled "
                          "yet; call after a frame has been advanced.");
        auto A        = impl.bcoo_A.cview();
        auto triplets = A.triplet_count();
        UIPC_ASSERT_THROW(block_rows.size() == triplets,
                          "LinearSystemAdjoint: {} triplets requested, the "
                          "system has {}",
                          block_rows.size(),
                          triplets);

        std::vector<int>       rows(triplets);
        std::vector<int>       cols(triplets);
        std::vector<Matrix3x3> values(triplets);
        A.row_indices().copy_to(rows.data());
        A.col_indices().copy_to(cols.data());
        A.values().copy_to(values.data());
        impl.b.buffer_view().copy_to(gradient.data());
        cuda_tool::wait_device();

        for(SizeT t = 0; t < triplets; ++t)
        {
            block_rows[t]     = rows[t];
            block_cols[t]     = cols[t];
            const auto& block = values[t];
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                    block_values[9 * t + 3 * i + j] = block(i, j);
        }
    }

    Float do_solve(span<const Float> rhs, span<Float> solution, Float rel_tol, SizeT max_rounds) override
    {
        auto& impl = m_system.m_impl;
        UIPC_ASSERT_THROW(!impl.empty_system,
                          "LinearSystemAdjoint: no system has been assembled "
                          "yet; call after a frame has been advanced.");
        auto dofs = impl.x.size();
        UIPC_ASSERT_THROW(rhs.size() == dofs && solution.size() == dofs,
                          "LinearSystemAdjoint: rhs has {} and solution {} "
                          "elements, the system has {} degrees of freedom",
                          rhs.size(),
                          solution.size(),
                          dofs);

        // Keep the frame's own gradient and Newton step.
        std::vector<Float> b_saved(dofs);
        std::vector<Float> x_saved(dofs);
        impl.b.buffer_view().copy_to(b_saved.data());
        impl.x.buffer_view().copy_to(x_saved.data());
        cuda_tool::wait_device();

        // Iterative refinement over the frame's solver, whose tolerance is
        // set for an inexact Newton step, not for an adjoint.
        Float rhs_norm = 0.0;
        for(auto v : rhs)
            rhs_norm += v * v;
        rhs_norm = std::sqrt(rhs_norm);
        std::vector<Float> residual(rhs.begin(), rhs.end());
        std::vector<Float> step(dofs);
        std::vector<Float> product(dofs);
        std::fill(solution.begin(), solution.end(), 0.0);
        Float relative = 1.0;
        if(rhs_norm > 0.0)
        {
            m_x.resize_discard(dofs);
            m_y.resize_discard(dofs);
            for(SizeT round = 0; round < max_rounds; ++round)
            {
                impl.b.buffer_view().copy_from(residual.data());
                cuda_tool::wait_device();
                impl.solve_linear_system();
                cuda_tool::wait_device();
                impl.x.buffer_view().copy_to(step.data());
                cuda_tool::wait_device();
                for(SizeT i = 0; i < dofs; ++i)
                    solution[i] += step[i];
                m_x.buffer_view().copy_from(solution.data());
                cuda_tool::wait_device();
                impl.spmv(1.0, m_x.cview(), 0.0, m_y.view());
                cuda_tool::wait_device();
                m_y.buffer_view().copy_to(product.data());
                cuda_tool::wait_device();
                Float residual_norm = 0.0;
                for(SizeT i = 0; i < dofs; ++i)
                {
                    residual[i] = rhs[i] - product[i];
                    residual_norm += residual[i] * residual[i];
                }
                relative = std::sqrt(residual_norm) / rhs_norm;
                UIPC_ASSERT_THROW(std::isfinite(relative),
                                  "LinearSystemAdjoint: the residual is not "
                                  "finite after refinement round {}",
                                  round);
                if(relative <= rel_tol)
                    break;
            }
        }
        else
        {
            relative = 0.0;
        }

        impl.b.buffer_view().copy_from(b_saved.data());
        impl.x.buffer_view().copy_from(x_saved.data());
        cuda_tool::wait_device();
        return relative;
    }

  private:
    GlobalLinearSystem&                 m_system;
    cuda_tool::DeviceDenseVector<Float> m_x;
    cuda_tool::DeviceDenseVector<Float> m_y;
};

// A SimSystem that registers the LinearSystemAdjointFeature with the engine.
class LinearSystemAdjoint final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    virtual void do_build() override
    {
        auto& system = require<GlobalLinearSystem>();
        auto overrider = std::make_shared<LinearSystemAdjointFeatureOverrider>(system);
        auto feature = std::make_shared<diff_sim::LinearSystemAdjointFeature>(overrider);
        features().insert(feature);
    }
};

REGISTER_SIM_SYSTEM(LinearSystemAdjoint);
}  // namespace uipc::backend::cuda
