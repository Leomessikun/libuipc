#include <pyuipc/diff_sim/module.h>
#include <pyuipc/diff_sim/linear_system_adjoint_feature.h>
namespace pyuipc::diff_sim
{
PyModule::PyModule(py::module& m)
{
    // PyParameterCollection is exported early in main module
    PyLinearSystemAdjointFeature{m};
}
}  // namespace pyuipc::diff_sim
