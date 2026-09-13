#include <app/app.h>
#include <uipc/uipc.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/constitution/soft_position_constraint.h>
#include <uipc/diff_sim/linear_system_adjoint_feature.h>
#include <Eigen/Dense>

// The assembled system of a frame, exported and solved against through the
// LinearSystemAdjointFeature: one tetrahedron under gravity with one vertex
// held by a soft position constraint.
TEST_CASE("linear_system_adjoint_feature", "[cuda][linear_system][diff_sim]")
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    std::string this_output_path =
        fmt::format("{}ipc/", AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE));

    Engine engine{"cuda", this_output_path};
    World  world{engine};

    auto config                         = Scene::default_config();
    config["gravity"]                   = Vector3{0, -9.8, 0};
    config["contact"]["enable"]         = false;
    config["linear_system"]["tol_rate"] = 1e-8;
    test::Scene::dump_config(config, this_output_path);

    Scene scene{config};
    {
        StableNeoHookean       snh;
        SoftPositionConstraint spc;
        auto                   object = scene.objects().create("tet");
        vector<Vector4i>       Ts     = {Vector4i{0, 1, 2, 3}};
        vector<Vector3>        Vs     = {Vector3{0, 1, 0},
                                         Vector3{0, 0, 1},
                                         Vector3{-std::sqrt(3) / 2, 0, -0.5},
                                         Vector3{std::sqrt(3) / 2, 0, -0.5}};
        auto                   mesh   = tetmesh(Vs, Ts);
        label_surface(mesh);
        label_triangle_orient(mesh);
        snh.apply_to(mesh, ElasticModuli::youngs_poisson(1e5, 0.499), 1e3);
        spc.apply_to(mesh, 100.0);
        auto is_constrained = mesh.vertices().find<IndexT>(builtin::is_constrained);
        REQUIRE(is_constrained);
        view(*is_constrained)[0] = 1;  // vertex 0 is held at its rest position
        object->geometries().create(mesh);
    }
    world.init(scene);
    REQUIRE(world.is_valid());

    auto feature = world.features().find<diff_sim::LinearSystemAdjointFeature>();
    REQUIRE(feature);

    for(int i = 0; i < 3; ++i)
    {
        world.advance();
        world.retrieve();
    }

    const SizeT dofs     = feature->dof_count();
    const SizeT triplets = feature->triplet_count();
    REQUIRE(dofs == 12);
    REQUIRE(triplets > 0);

    vector<IndexT> rows(triplets);
    vector<IndexT> cols(triplets);
    vector<Float>  values(9 * triplets);
    vector<Float>  gradient(dofs);
    feature->export_system(rows, cols, values, gradient);

    // Assemble the dense symmetric matrix from the upper block triangle.
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dofs, dofs);
    for(SizeT t = 0; t < triplets; ++t)
    {
        REQUIRE(rows[t] >= 0);
        REQUIRE(cols[t] >= 0);
        REQUIRE(rows[t] <= cols[t]);
        REQUIRE(3 * cols[t] + 3 <= IndexT(dofs));
        Eigen::Matrix3d block;
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
            {
                block(i, j) = values[9 * t + 3 * i + j];
                REQUIRE(std::isfinite(block(i, j)));
            }
        H.block<3, 3>(3 * rows[t], 3 * cols[t]) += block;
        if(rows[t] != cols[t])
            H.block<3, 3>(3 * cols[t], 3 * rows[t]) += block.transpose();
    }
    REQUIRE((H - H.transpose()).norm() <= 1e-12 * H.norm());
    for(auto g : gradient)
        REQUIRE(std::isfinite(g));

    // A solve against the frame's system agrees with the exported matrix.
    vector<Float> rhs(dofs);
    for(SizeT i = 0; i < dofs; ++i)
        rhs[i] = std::sin(0.7 * Float(i) + 0.3);
    vector<Float> solution(dofs);
    Float         reached = feature->solve(rhs, solution, 1e-8, 32);
    REQUIRE(reached <= 1e-8);
    Eigen::Map<Eigen::VectorXd> x(solution.data(), dofs);
    Eigen::Map<Eigen::VectorXd> b(rhs.data(), dofs);
    Eigen::VectorXd             residual = H * x - b;
    // The refinement reports its residual against the backend's own SpMV; the
    // exported matrix must agree with that operator.
    REQUIRE(residual.norm() <= 1e-6 * b.norm());

    // The frame's own gradient is restored after a solve, and a second solve
    // gives the same answer.
    vector<Float> gradient_after(dofs);
    feature->export_system(rows, cols, values, gradient_after);
    for(SizeT i = 0; i < dofs; ++i)
        REQUIRE(gradient_after[i] == gradient[i]);
    vector<Float> solution_again(dofs);
    feature->solve(rhs, solution_again, 1e-8, 32);
    Eigen::Map<Eigen::VectorXd> x2(solution_again.data(), dofs);
    REQUIRE((x2 - x).norm() <= 1e-9 * (x.norm() + 1e-300));

    // Wrong sizes are refused.
    vector<Float> short_rhs(dofs - 1);
    REQUIRE_THROWS(feature->solve(short_rhs, solution));

    // Advancing still works afterwards.
    world.advance();
    world.retrieve();
    REQUIRE(world.is_valid());
}
