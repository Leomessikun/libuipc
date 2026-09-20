#include <app/app.h>
#include <uipc/uipc.h>
#include <uipc/constitution/strain_limiting_baraff_witkin.h>
#include <uipc/constitution/neo_hookean_shell.h>

// The strain-limiting Baraff-Witkin shell is the cloth model this project's
// dressing work runs on, and it had no end-to-end simulation coverage: only a
// construction test in `apps/tests/core`. This hangs a strip from two corners
// under gravity and checks the property the model exists for, by running the
// same mesh and the same load under the unlimited Neo-Hookean shell: the
// limited one must not stretch further than the unlimited one.
namespace
{
using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;

constexpr SizeT k_rows   = 12;
constexpr SizeT k_frames = 60;
// Soft and heavy on purpose: at cloth-like stiffness the strip barely
// stretches and the limit never engages, so the two models are
// indistinguishable and the test would assert nothing.
constexpr Float k_youngs  = 2.0;    // Pa
constexpr Float k_density = 5.0e3;  // kg/m^3

// A flat strip in the xz plane, hung by the two corners of its first row.
void build_strip(vector<Vector3>& Vs, vector<Vector3i>& Fs)
{
    constexpr Float step = 0.05;
    for(SizeT i = 0; i < k_rows; ++i)
        for(SizeT j = 0; j < 2; ++j)
            Vs.push_back(
                Vector3{static_cast<Float>(j) * step, 0.5, static_cast<Float>(i) * step});
    for(SizeT i = 0; i + 1 < k_rows; ++i)
    {
        IndexT a = static_cast<IndexT>(2 * i);
        Fs.push_back(Vector3i{a, a + 1, a + 2});
        Fs.push_back(Vector3i{a + 1, a + 3, a + 2});
    }
}

// The largest edge length divided by its rest length, over the whole mesh.
Float max_edge_stretch(span<const Vector3> rest, span<const Vector3> now, span<const Vector3i> Fs)
{
    Float worst = 0.0;
    for(const auto& f : Fs)
    {
        for(int k = 0; k < 3; ++k)
        {
            IndexT a        = f[k];
            IndexT b        = f[(k + 1) % 3];
            Float  rest_len = (rest[a] - rest[b]).norm();
            if(rest_len <= 0.0)
                continue;
            worst = std::max(worst, (now[a] - now[b]).norm() / rest_len);
        }
    }
    return worst;
}

// Runs the strip with one shell constitution and returns its worst stretch.
// `limited` selects the strain-limiting model; otherwise the Neo-Hookean one,
// with the same moduli, density and thickness, so only the model differs.
Float run_strip(bool limited, std::string_view output_path)
{
    vector<Vector3>  Vs;
    vector<Vector3i> Fs;
    build_strip(Vs, Fs);
    const vector<Vector3> rest = Vs;

    Engine engine{"cuda", std::string{output_path}};
    World  world{engine};

    auto config                             = test::Scene::default_config();
    config["gravity"]                       = Vector3{0, -9.8, 0};
    config["contact"]["enable"]             = false;
    config["contact"]["friction"]["enable"] = false;

    Scene                    scene{config};
    S<SimplicialComplexSlot> slot;
    {
        auto object = scene.objects().create("strip");
        auto mesh   = trimesh(Vs, Fs);
        label_surface(mesh);

        auto moduli = ElasticModuli2D::youngs_poisson(k_youngs, 0.49);
        if(limited)
        {
            StrainLimitingBaraffWitkinShell slbw;
            slbw.apply_to(mesh, moduli, k_density);
        }
        else
        {
            NeoHookeanShell nhs;
            nhs.apply_to(mesh, moduli, k_density);
        }

        auto is_fixed      = mesh.vertices().find<IndexT>(builtin::is_fixed);
        auto is_fixed_view = view(*is_fixed);
        is_fixed_view[0]   = 1;
        is_fixed_view[1]   = 1;

        auto [created, rest_slot] = object->geometries().create(mesh);
        slot                      = created;
    }

    world.init(scene);
    REQUIRE(world.is_valid());

    while(world.frame() < k_frames)
    {
        world.advance();
        REQUIRE(world.is_valid());
        world.retrieve();
    }

    auto positions = slot->geometry().positions().view();
    REQUIRE(positions.size() == rest.size());
    for(const auto& p : positions)
        REQUIRE(p.allFinite());
    return max_edge_stretch(rest, positions, Fs);
}
}  // namespace

TEST_CASE("94_strain_limiting_baraff_witkin_shell", "[fem][cloth]")
{
    auto output_path = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);

    Float limited = run_strip(true, fmt::format("{}limited/", output_path));
    Float unlimited = run_strip(false, fmt::format("{}unlimited/", output_path));

    INFO("strain-limited max stretch " << limited << ", Neo-Hookean " << unlimited);
    // The comparison is only meaningful if the unlimited model actually
    // stretches. At cloth-like stiffness neither does - both sit at 1.0002 -
    // and the test would pass without asserting anything. Under this soft,
    // heavy strip the observed values are 6.77 limited against 35.82
    // unlimited, so the bound below has a wide margin.
    REQUIRE(unlimited > 2.0);
    REQUIRE(limited >= 1.0);
    REQUIRE(limited <= unlimited + 1e-6);
}
