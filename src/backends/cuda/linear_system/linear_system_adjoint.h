#pragma once
#include <sim_system.h>
#include <uipc/diff_sim/linear_system_adjoint_feature.h>

namespace uipc::backend::cuda
{
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

  protected:
    virtual void do_build() override;

  private:
    ExportMode m_export_mode = ExportMode::LastIterate;
};
}  // namespace uipc::backend::cuda
