#include <uipc/diff_sim/linear_system_adjoint_feature.h>
#include <uipc/common/log.h>

namespace uipc::diff_sim
{
LinearSystemAdjointFeature::LinearSystemAdjointFeature(S<LinearSystemAdjointFeatureOverrider> overrider)
    : m_impl(std::move(overrider))
{
    UIPC_ASSERT_THROW(m_impl, "LinearSystemAdjointFeatureOverrider must not be null.");
}

SizeT LinearSystemAdjointFeature::dof_count() const
{
    return m_impl->get_dof_count();
}

SizeT LinearSystemAdjointFeature::triplet_count() const
{
    return m_impl->get_triplet_count();
}

void LinearSystemAdjointFeature::set_export_mode(LinearSystemExportMode mode)
{
    auto value = static_cast<int>(mode);
    UIPC_ASSERT_THROW(value >= 0 && value <= 2, "set_export_mode: unknown export mode {}", value);
    m_impl->do_set_export_mode(mode);
}

LinearSystemExportMode LinearSystemAdjointFeature::export_mode() const
{
    return m_impl->do_export_mode();
}

void LinearSystemAdjointFeature::export_system(span<IndexT> block_rows,
                                               span<IndexT> block_cols,
                                               span<Float>  block_values,
                                               span<Float>  gradient) const
{
    auto triplets = triplet_count();
    auto dofs     = dof_count();
    UIPC_ASSERT_THROW(block_rows.size() == triplets && block_cols.size() == triplets,
                      "export_system: block index spans have {} and {} elements, "
                      "the system has {} triplets",
                      block_rows.size(),
                      block_cols.size(),
                      triplets);
    UIPC_ASSERT_THROW(block_values.size() == 9 * triplets,
                      "export_system: block value span has {} elements, "
                      "expected 9 * {} = {}",
                      block_values.size(),
                      triplets,
                      9 * triplets);
    UIPC_ASSERT_THROW(gradient.size() == dofs,
                      "export_system: gradient span has {} elements, the "
                      "system has {} degrees of freedom",
                      gradient.size(),
                      dofs);
    m_impl->do_export_system(block_rows, block_cols, block_values, gradient);
}

Float LinearSystemAdjointFeature::solve(span<const Float> rhs,
                                        span<Float>       solution,
                                        Float             rel_tol,
                                        SizeT             max_rounds) const
{
    auto dofs = dof_count();
    UIPC_ASSERT_THROW(rel_tol > 0.0 && max_rounds >= 1,
                      "solve: rel_tol must be positive and max_rounds at least 1, got {} and {}",
                      rel_tol,
                      max_rounds);
    UIPC_ASSERT_THROW(rhs.size() == dofs && solution.size() == dofs,
                      "solve: rhs has {} and solution {} elements, the "
                      "system has {} degrees of freedom",
                      rhs.size(),
                      solution.size(),
                      dofs);
    return m_impl->do_solve(rhs, solution, rel_tol, max_rounds);
}

std::string_view LinearSystemAdjointFeature::get_name() const
{
    return FeatureName;
}
}  // namespace uipc::diff_sim
