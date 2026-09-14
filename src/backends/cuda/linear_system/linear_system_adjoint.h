#pragma once
#include <sim_system.h>
#include <uipc/diff_sim/linear_system_adjoint_feature.h>

namespace uipc::backend::cuda
{
class VertexHalfPlaneFrictionalContact;

/**
 * @brief Registers the `LinearSystemAdjointFeature` and keeps its export
 * mode; the engine reads the mode after every Newton loop to decide whether
 * to re-assemble the system at the accepted state (see advance_ipc.cu).
 */
class LinearSystemAdjoint final : public SimSystem
{
  public:
    using SimSystem::SimSystem;
    using ExportMode = diff_sim::LinearSystemExportMode;

    ExportMode export_mode() const noexcept { return m_export_mode; }
    void set_export_mode(ExportMode mode) noexcept { m_export_mode = mode; }

    /// After the engine's re-assembly at the accepted state: compute the
    /// friction pairs' dG/dx_prev blocks (the lagged coupling the adjoint
    /// chain needs beyond inertia). Invalidated by every advance.
    void after_reassembly();
    void invalidate_coupling() noexcept { m_coupling_valid = false; }
    bool coupling_valid() const noexcept { return m_coupling_valid; }
    VertexHalfPlaneFrictionalContact* half_plane_friction() const noexcept
    {
        return m_half_plane_friction;
    }

  protected:
    virtual void do_build() override;

  private:
    ExportMode                        m_export_mode = ExportMode::LastIterate;
    VertexHalfPlaneFrictionalContact* m_half_plane_friction = nullptr;
    bool                              m_coupling_valid      = false;
};
}  // namespace uipc::backend::cuda
