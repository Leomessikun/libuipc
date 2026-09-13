# World-level tests of the differentiable-simulation features. Like sim_case,
# the backend is loaded as a module: linking the CUDA backend objects into the
# test binary as backend_cuda does gives every kernel a second copy and breaks
# launches once an Engine loads the module.
file(GLOB SOURCE CONFIGURE_DEPENDS "*.cpp" "*.h")
uipc_add_test(diff_sim ${SOURCE})
target_link_libraries(diff_sim PRIVATE uipc::backends)
