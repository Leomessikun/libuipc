#pragma once
#include <uipc/core/feature.h>
#include <uipc/common/span.h>
#include <uipc/common/type_define.h>

namespace uipc::diff_sim
{
/**
 * @brief Backend-side access to the assembled linear system of the current
 * frame, for adjoint and tangent sensitivities.
 *
 * After `World::advance()` the backend still holds the system it assembled in
 * the frame's last Newton iteration: the (projected) Hessian `H` as a block
 * sparse matrix and the gradient `g` it solved against. The overrider exports
 * that system to host memory and solves `H x = rhs` for a caller's right-hand
 * side with the backend's own iterative solver and preconditioner, leaving
 * the frame's own solution untouched.
 *
 * The export mode chooses which system the next frames leave behind:
 * `LastIterate` (the default) is the matrix of the frame's last Newton
 * iteration, as the solver used it; `Converged` re-assembles the contact
 * pairs, gradient and projected Hessian at the accepted state after Newton
 * ends; `ConvergedRaw` does the same without projecting the stencil Hessians
 * to positive semidefinite, which is the Jacobian the implicit function
 * theorem needs. The forward solve is never affected.
 */
enum class LinearSystemExportMode : int
{
    LastIterate  = 0,
    Converged    = 1,
    ConvergedRaw = 2,
};

class UIPC_CORE_API LinearSystemAdjointFeatureOverrider
{
  public:
    virtual ~LinearSystemAdjointFeatureOverrider() = default;

    virtual void do_set_export_mode(LinearSystemExportMode mode) = 0;
    virtual LinearSystemExportMode do_export_mode()              = 0;

    /// Number of lagged coupling blocks the last re-assembly computed (0 in
    /// `LastIterate` mode or without friction).
    virtual SizeT get_prev_coupling_count() = 0;
    /**
     * @param rows [out] `prev_coupling_count` global vertex indices
     * @param cols [out] the same count: the previous-position vertex each
     *        block differentiates against (the pair's own vertex for a
     *        half-plane contact)
     * @param blocks [out] `9 * count` scalars, row-major 3x3 blocks of
     *        dG/dx_prev in the system's scaling
     */
    virtual void do_export_prev_coupling(span<IndexT> rows,
                                         span<IndexT> cols,
                                         span<Float>  blocks) = 0;

    /// Number of scalar degrees of freedom of the assembled system.
    virtual SizeT get_dof_count() = 0;
    /// Number of 3x3 blocks stored (the upper block triangle, diagonal
    /// blocks whole, duplicates summed).
    virtual SizeT get_triplet_count() = 0;
    /**
     * @param block_rows [out] `triplet_count` block row indices
     * @param block_cols [out] `triplet_count` block column indices
     * @param block_values [out] `9 * triplet_count` scalars, one row-major
     *        3x3 block per triplet
     * @param gradient [out] `dof_count` scalars, the gradient the frame's
     *        last Newton iteration solved against
     */
    virtual void do_export_system(span<IndexT> block_rows,
                                  span<IndexT> block_cols,
                                  span<Float>  block_values,
                                  span<Float>  gradient) = 0;
    /**
     * @brief Solve `H x = rhs` with the current system to a relative
     * residual, by iterative refinement over the frame's own (inexact)
     * solver: `x += solve(rhs - H x)` until `|rhs - H x| <= rel_tol |rhs|`
     * or `max_rounds` rounds.
     * @param rhs [in] `dof_count` scalars
     * @param solution [out] `dof_count` scalars
     * @return the relative residual reached
     */
    virtual Float do_solve(span<const Float> rhs,
                           span<Float>       solution,
                           Float             rel_tol,
                           SizeT             max_rounds) = 0;
};

class UIPC_CORE_API LinearSystemAdjointFeature final : public core::Feature
{
  public:
    constexpr static std::string_view FeatureName = "diff_sim/linear_system_adjoint";

    LinearSystemAdjointFeature(S<LinearSystemAdjointFeatureOverrider> overrider);

    SizeT dof_count() const;
    SizeT triplet_count() const;

    /**
     * @brief Choose the system the following frames leave for export; takes
     * effect from the next `World::advance()`. `ConvergedRaw` may be
     * indefinite, so `solve()` refuses it: export it and factorise on the host.
     */
    void                   set_export_mode(LinearSystemExportMode mode);
    LinearSystemExportMode export_mode() const;

    /**
     * @brief The explicit dependence of the frame's friction gradient on the
     * previous substep's positions, `dG_f/dx_prev`, one 3x3 block per
     * friction pair, computed by the `Converged` and `ConvergedRaw` modes'
     * re-assembly at the accepted state (friction's lagged normal force and
     * relative displacement). Zero blocks in `LastIterate` mode.
     */
    SizeT prev_coupling_count() const;
    void export_prev_coupling(span<IndexT> rows, span<IndexT> cols, span<Float> blocks) const;

    /**
     * @brief Copy the assembled system of the current frame to host memory.
     *
     * The spans must have exactly `triplet_count()`, `triplet_count()`,
     * `9 * triplet_count()` and `dof_count()` elements.
     */
    void export_system(span<IndexT> block_rows,
                       span<IndexT> block_cols,
                       span<Float>  block_values,
                       span<Float>  gradient) const;

    /**
     * @brief Solve `H x = rhs` with the current frame's assembled system.
     *
     * Both spans must have `dof_count()` elements. The frame's solver is
     * run repeatedly on the residual (iterative refinement) until the
     * relative residual is at most `rel_tol` or `max_rounds` rounds have
     * been spent; the frame's own gradient and solution are restored
     * afterwards.
     * @return the relative residual `|rhs - H x| / |rhs|` reached
     */
    Float solve(span<const Float> rhs,
                span<Float>       solution,
                Float             rel_tol    = 1e-6,
                SizeT             max_rounds = 32) const;

  private:
    virtual std::string_view               get_name() const final override;
    S<LinearSystemAdjointFeatureOverrider> m_impl;
};
}  // namespace uipc::diff_sim
