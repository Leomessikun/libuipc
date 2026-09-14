#include <app/app.h>
#include <uipc/uipc.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/constitution/soft_position_constraint.h>
#include <uipc/diff_sim/linear_system_adjoint_feature.h>
#include <Eigen/Dense>

namespace
{
constexpr uipc::Float TetDensity = 1e3;

// One tetrahedron whose vertex 0 is held at its rest position by a soft
// position constraint of the given strength.
void make_held_tet(uipc::core::Scene& scene, uipc::Float strength)
{
    using namespace uipc;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

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
    snh.apply_to(mesh, ElasticModuli::youngs_poisson(1e5, 0.499), TetDensity);
    spc.apply_to(mesh, strength);
    auto is_constrained = mesh.vertices().find<IndexT>(builtin::is_constrained);
    REQUIRE(is_constrained);
    view(*is_constrained)[0] = 1;
    object->geometries().create(mesh);
}

// Lumped mass of one vertex of that tetrahedron: density × volume / 4.
uipc::Float held_vertex_mass()
{
    using namespace uipc;
    Vector3 a{0, 1, 0};
    Vector3 b{0, 0, 1};
    Vector3 c{-std::sqrt(3) / 2, 0, -0.5};
    Vector3 d{std::sqrt(3) / 2, 0, -0.5};
    Float   volume = std::abs((b - a).dot((c - a).cross(d - a))) / 6.0;
    return TetDensity * volume / 4.0;
}
}  // namespace

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
    make_held_tet(scene, 100.0);
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

// The held vertex's diagonal block grows by strength × lumped mass: the soft
// position constraint kernel writes s·m·I and the assembly adds it once, with
// no time-step scaling. Without gravity both worlds sit at rest, so the
// elastic and inertial parts of the block are identical and cancel.
TEST_CASE("linear_system_adjoint_constraint_stiffness", "[cuda][linear_system][diff_sim]")
{
    using namespace uipc;
    using namespace uipc::core;

    std::string this_output_path =
        fmt::format("{}stiffness/", AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE));

    auto held_block = [&](Float strength) -> Eigen::Matrix3d
    {
        Engine engine{"cuda", this_output_path};
        World  world{engine};

        auto config                         = Scene::default_config();
        config["gravity"]                   = Vector3{0, 0, 0};
        config["contact"]["enable"]         = false;
        config["linear_system"]["tol_rate"] = 1e-8;

        Scene scene{config};
        make_held_tet(scene, strength);
        world.init(scene);
        REQUIRE(world.is_valid());
        auto feature = world.features().find<diff_sim::LinearSystemAdjointFeature>();
        REQUIRE(feature);
        world.advance();
        world.retrieve();

        const SizeT    dofs     = feature->dof_count();
        const SizeT    triplets = feature->triplet_count();
        vector<IndexT> rows(triplets);
        vector<IndexT> cols(triplets);
        vector<Float>  values(9 * triplets);
        vector<Float>  gradient(dofs);
        feature->export_system(rows, cols, values, gradient);

        Eigen::Matrix3d block = Eigen::Matrix3d::Zero();
        bool            found = false;
        for(SizeT t = 0; t < triplets; ++t)
        {
            if(rows[t] != 0 || cols[t] != 0)
                continue;
            found = true;
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                    block(i, j) += values[9 * t + 3 * i + j];
        }
        REQUIRE(found);
        return block;
    };

    Eigen::Matrix3d difference = held_block(200.0) - held_block(100.0);
    Eigen::Matrix3d expected = 100.0 * held_vertex_mass() * Eigen::Matrix3d::Identity();
    INFO("measured stiffness per unit strength and mass: "
         << difference.trace() / (3.0 * 100.0 * held_vertex_mass()));
    REQUIRE((difference - expected).norm() <= 1e-6 * expected.norm());
}
