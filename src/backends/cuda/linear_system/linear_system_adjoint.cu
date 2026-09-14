#include <sim_system.h>
#include <linear_system/global_linear_system.h>
#include <linear_system/linear_system_adjoint.h>
#include <contact_system/vertex_half_plane_frictional_contact.h>
#include <utils/make_spd.h>
#include <uipc/diff_sim/linear_system_adjoint_feature.h>
#include <uipc/common/log.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace uipc::backend::cuda
{
// The projection switch every make_spd() reads (declared in utils/make_spd.h).
__device__ int uipc_hessian_projection_enabled = 1;
static bool    host_hessian_projection_enabled = true;

void set_hessian_projection(bool enabled)
{
    int value = enabled ? 1 : 0;
    cuda_tool::wait_device();
    CUDA_TOOL_CHECK(cudaMemcpyToSymbol(uipc_hessian_projection_enabled, &value, sizeof(int)));
    host_hessian_projection_enabled = enabled;
}

bool hessian_projection_enabled()
{
    return host_hessian_projection_enabled;
}

/**
 * @brief Exposes the assembled system of the current frame and a solve
 * against it, for sensitivities computed outside the backend.
 *
 * `GlobalLinearSystem` keeps `bcoo_A` (the block-sparse Hessian after
 * symmetric compression, upper block triangle), `b` (the gradient) and `x`
 * (the last Newton step) from the frame's last iteration until the next
 * frame's first solve, or, in the `Converged` and `ConvergedRaw` export
 * modes, the system the engine re-assembled at the accepted state after the
 * Newton loop. Nothing here rebuilds the system; `do_solve` swaps a caller's
 * right-hand side in, runs the frame's solver on the frame's matrix and
 * preconditioner, and swaps the frame's own vectors back.
 */
class LinearSystemAdjointFeatureOverrider final : public diff_sim::LinearSystemAdjointFeatureOverrider
{
  public:
    LinearSystemAdjointFeatureOverrider(GlobalLinearSystem& system, LinearSystemAdjoint& owner)
        : m_system(system)
        , m_owner(owner)
    {
    }

    void do_set_export_mode(diff_sim::LinearSystemExportMode mode) override
    {
        m_owner.set_export_mode(mode);
    }

    diff_sim::LinearSystemExportMode do_export_mode() override
    {
        return m_owner.export_mode();
    }

    SizeT get_prev_coupling_count() override
    {
        auto* friction = m_owner.half_plane_friction();
        if(!m_owner.coupling_valid() || !friction)
            return 0;
        return friction->prev_coupling().size();
    }

    void do_export_prev_coupling(span<IndexT> rows, span<IndexT> cols, span<Float> blocks) override
    {
        auto count = get_prev_coupling_count();
        UIPC_ASSERT_THROW(rows.size() == count && cols.size() == count
                              && blocks.size() == 9 * count,
                          "LinearSystemAdjoint: {} coupling blocks requested, "
                          "the frame has {}",
                          rows.size(),
                          count);
        if(count == 0)
            return;
        auto*                  friction = m_owner.half_plane_friction();
        std::vector<Vector2i>  pairs(count);
        std::vector<Matrix3x3> values(count);
        friction->PHs().copy_to(pairs.data());
        friction->prev_coupling().copy_to(values.data());
        cuda_tool::wait_device();
        for(SizeT t = 0; t < count; ++t)
        {
            rows[t] = pairs[t](0);
            cols[t] = pairs[t](0);
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                    blocks[9 * t + 3 * i + j] = values[t](i, j);
        }
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
        UIPC_ASSERT_THROW(m_owner.export_mode() != diff_sim::LinearSystemExportMode::ConvergedRaw,
                          "LinearSystemAdjoint: the raw system may be "
                          "indefinite and has no preconditioner; export it "
                          "and factorise on the host instead of solve().");
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
    LinearSystemAdjoint&                m_owner;
    cuda_tool::DeviceDenseVector<Float> m_x;
    cuda_tool::DeviceDenseVector<Float> m_y;
};

void LinearSystemAdjoint::after_reassembly()
{
    m_coupling_valid = false;
    if(m_half_plane_friction)
    {
        m_half_plane_friction->compute_prev_coupling();
        cuda_tool::wait_device();
        m_coupling_valid = true;
    }
}

void LinearSystemAdjoint::do_build()
{
    auto& system = require<GlobalLinearSystem>();
    // The registered model is a subclass; a compatible lookup finds it.
    m_half_plane_friction = find<VertexHalfPlaneFrictionalContact>({.exact = false});
    auto overrider = std::make_shared<LinearSystemAdjointFeatureOverrider>(system, *this);
    auto feature = std::make_shared<diff_sim::LinearSystemAdjointFeature>(overrider);
    features().insert(feature);
}

REGISTER_SIM_SYSTEM(LinearSystemAdjoint);
}  // namespace uipc::backend::cuda
