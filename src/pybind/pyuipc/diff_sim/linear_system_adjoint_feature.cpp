#include <pyuipc/diff_sim/linear_system_adjoint_feature.h>
#include <uipc/diff_sim/linear_system_adjoint_feature.h>
#include <uipc/core/feature.h>
#include <pybind11/numpy.h>

namespace pyuipc::diff_sim
{
using namespace uipc::diff_sim;
using uipc::Float;
using uipc::IndexT;
using uipc::SizeT;
using uipc::core::IFeature;

PyLinearSystemAdjointFeature::PyLinearSystemAdjointFeature(py::module& m)
{
    auto class_LinearSystemAdjointFeature =
        py::class_<LinearSystemAdjointFeature, IFeature, S<LinearSystemAdjointFeature>>(
            m,
            "LinearSystemAdjointFeature",
            R"(The assembled linear system of the current frame: export it, or solve
against it with the backend's own solver. Valid after `World.advance()` and
until the next frame's first Newton solve.)");

    class_LinearSystemAdjointFeature.def("dof_count",
                                         &LinearSystemAdjointFeature::dof_count,
                                         R"(Number of scalar degrees of freedom of the assembled system.)");

    class_LinearSystemAdjointFeature.def(
        "triplet_count",
        &LinearSystemAdjointFeature::triplet_count,
        R"(Number of 3x3 blocks stored (upper block triangle, diagonal blocks whole).)");

    class_LinearSystemAdjointFeature.def(
        "export_system",
        [](const LinearSystemAdjointFeature& self) -> py::tuple
        {
            auto                triplets = self.triplet_count();
            auto                dofs     = self.dof_count();
            py::array_t<IndexT> rows{py::ssize_t(triplets)};
            py::array_t<IndexT> cols{py::ssize_t(triplets)};
            py::array_t<Float>  values(
                {py::ssize_t(triplets), py::ssize_t(3), py::ssize_t(3)});
            py::array_t<Float> gradient{py::ssize_t(dofs)};
            self.export_system(uipc::span<IndexT>{rows.mutable_data(), triplets},
                               uipc::span<IndexT>{cols.mutable_data(), triplets},
                               uipc::span<Float>{values.mutable_data(), 9 * triplets},
                               uipc::span<Float>{gradient.mutable_data(), dofs});
            return py::make_tuple(rows, cols, values, gradient);
        },
        R"(Copy the current frame's assembled system to host memory.

Returns:
    tuple: (block_rows[int32, T], block_cols[int32, T], block_values[float64, (T, 3, 3)],
    gradient[float64, dofs]) with T = triplet_count(). Only blocks with row <= col are
    stored; a consumer mirrors the off-diagonal blocks. Fixed degrees of freedom carry
    identity rows.)");

    class_LinearSystemAdjointFeature.def(
        "solve",
        [](const LinearSystemAdjointFeature& self,
           py::array_t<Float>                rhs,
           Float                             rel_tol,
           SizeT                             max_rounds) -> py::tuple
        {
            auto dofs = self.dof_count();
            auto buf  = rhs.request();
            if(buf.ndim != 1 || SizeT(buf.shape[0]) != dofs)
                throw std::runtime_error("solve: rhs must be a 1-D float64 array of dof_count() elements");
            py::array_t<Float> rhs_c = py::array_t<Float>::ensure(rhs);
            py::array_t<Float> solution{py::ssize_t(dofs)};
            Float residual = self.solve(uipc::span<const Float>{rhs_c.data(), dofs},
                                        uipc::span<Float>{solution.mutable_data(), dofs},
                                        rel_tol,
                                        max_rounds);
            return py::make_tuple(solution, residual);
        },
        py::arg("rhs"),
        py::arg("rel_tol")    = 1e-6,
        py::arg("max_rounds") = 32,
        R"(Solve H x = rhs with the current frame's matrix and preconditioner.

Iterative refinement over the frame's own solver until the relative residual is at most
rel_tol or max_rounds rounds have been spent.

Args:
    rhs: float64 array of dof_count() elements.
    rel_tol: relative residual to reach (default 1e-6).
    max_rounds: refinement rounds at most (default 32).
Returns:
    tuple: (solution[float64, dofs], relative_residual). The frame's own gradient and
    Newton step are restored afterwards.)");

    class_LinearSystemAdjointFeature.attr("FeatureName") =
        LinearSystemAdjointFeature::FeatureName;
}
}  // namespace pyuipc::diff_sim
