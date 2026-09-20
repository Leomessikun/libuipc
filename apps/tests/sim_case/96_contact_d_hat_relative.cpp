#include <app/app.h>
#include <uipc/uipc.h>
#include <uipc/constitution/stable_neo_hookean.h>

// `contact/d_hat_relative` is documented as "positive values replace d_hat with
// this fraction of the rest-scene diagonal", and nothing selected it. The
// contract has two halves and both are checked here without depending on how
// the backend defines its diagonal:
//
//   * when it is positive it *replaces* the absolute value, so two scenes that
//     differ only in `contact/d_hat` must behave identically;
//   * when it is zero the absolute value is still in force, so the same two
//     scenes must behave differently.
//
// The second half is what stops the first from passing vacuously on a scene
// where d_hat happens not to matter.
namespace
{
using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;

constexpr SizeT k_frames = 30;

// A tetrahedron dropped onto the ground; only the contact gap is varied.
vector<Vector3> run(Float d_hat, Float d_hat_relative, std::string_view output_path)
{
    Engine engine{"cuda", std::string{output_path}};
    World  world{engine};

    auto config                             = test::Scene::default_config();
    config["gravity"]                       = Vector3{0, -9.8, 0};
    config["contact"]["enable"]             = true;
    config["contact"]["friction"]["enable"] = false;
    config["contact"]["d_hat"]              = d_hat;
    config["contact"]["d_hat_relative"]     = d_hat_relative;

    Scene                    scene{config};
    S<SimplicialComplexSlot> slot;
    {
        scene.contact_tabular().default_model(0.5, 1.0_GPa);
        auto default_contact = scene.contact_tabular().default_element();
        auto object          = scene.objects().create("tet");

        vector<Vector4i> Ts = {Vector4i{0, 1, 2, 3}};
        vector<Vector3>  Vs = {Vector3{0, 1, 0},
                               Vector3{0, 0, 1},
                               Vector3{-std::sqrt(3) / 2, 0, -0.5},
                               Vector3{std::sqrt(3) / 2, 0, -0.5}};
        std::transform(Vs.begin(),
                       Vs.end(),
                       Vs.begin(),
                       [&](auto& v) { return v * 0.3 + Vector3::UnitY() * 0.2; });

        auto mesh = tetmesh(Vs, Ts);
        label_surface(mesh);
        label_triangle_orient(mesh);

        // A finite element body, not an affine one: an affine body's motion
        // lives in its transform and its vertex positions never move, so a
        // position comparison would read the same rest mesh in every run.
        StableNeoHookean snh;
        snh.apply_to(mesh, ElasticModuli::youngs_poisson(10.0_MPa, 0.49));
        default_contact.apply_to(mesh);

        auto [created, rest_slot] = object->geometries().create(mesh);
        slot                      = created;

        ImplicitGeometry half_plane = ground(0.0);
        object->geometries().create(half_plane);
    }

    world.init(scene);
    REQUIRE(world.is_valid());
    while(world.frame() < k_frames)
    {
        world.advance();
        REQUIRE(world.is_valid());
        world.retrieve();
    }

    auto            positions = slot->geometry().positions().view();
    vector<Vector3> out{positions.begin(), positions.end()};
    for(const auto& p : out)
        REQUIRE(p.allFinite());
    return out;
}

Float largest_gap(span<const Vector3> a, span<const Vector3> b)
{
    REQUIRE(a.size() == b.size());
    Float worst = 0.0;
    for(SizeT i = 0; i < a.size(); ++i)
        worst = std::max(worst, (a[i] - b[i]).norm());
    return worst;
}
}  // namespace

TEST_CASE("96_contact_d_hat_relative", "[abd][contact]")
{
    auto out = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);

    // With the relative gap off, the absolute one decides.
    auto  absolute_small = run(0.002, 0.0, fmt::format("{}abs_small/", out));
    auto  absolute_large = run(0.020, 0.0, fmt::format("{}abs_large/", out));
    Float absolute_gap   = largest_gap(absolute_small, absolute_large);
    INFO("absolute d_hat changes the result by " << absolute_gap << " m");
    REQUIRE(absolute_gap > 1e-4);

    // With it on, the absolute one is replaced and no longer matters.
    auto relative_from_small = run(0.002, 0.01, fmt::format("{}rel_small/", out));
    auto relative_from_large = run(0.020, 0.01, fmt::format("{}rel_large/", out));
    Float relative_gap = largest_gap(relative_from_small, relative_from_large);
    INFO("with d_hat_relative set, absolute d_hat changes it by " << relative_gap << " m");
    REQUIRE(relative_gap < 1e-12);
}
