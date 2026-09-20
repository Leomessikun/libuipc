# What the IPC solver's tests do and do not cover — 2026-09-20

Status: audit complete, three tests added, one existing test found defective. The
suite is 98 simulation cases plus 106 other cases; every one passes except the
intermittent failure recorded below.

## Baseline before the audit

| target | cases | assertions |
|---|---:|---:|
| `uipc.sim_case` | 95 | 14,213 |
| `uipc.core` | 36 | 1,040 |
| `uipc.geometry` | 38 | 270 |
| `uipc.backend_cuda` | 22 | 337 |
| `uipc.diff_sim` | 3 | 563 |
| `uipc.sanity_check` | 3 | 100 |
| `uipc.common` | 3 | 11 |
| `uipc.regression` | 1 | 4 |

`ctest` reports all eight green in 70 s on this machine, which is what made the
defect below invisible: it fails about half the time and the suite is run once.

## What the audit asked

`scene_default_config.cpp` is the authoritative capability contract (ADR 0003), so
every key in it is a capability and the question is which are exercised. Of its 40
`add(...)` keys, 15 are named in no test. Four are debug dumps. The rest are solver
semantics: `linear_system/solver`, `linear_system/check_interval`,
`contact/d_hat_relative`, `contact/eps_velocity`, `contact/eps_velocity_relative`,
`contact/adaptive/kappa_eval_scale`, `contact/adaptive/max_kappa`, `newton/ccd_tol`,
`newton/transrate_tol`, `newton/velocity_tol_relative` and
`newton/semi_implicit/beta_tol`.

`apps/tests/core/scene_config.cpp` already checks the schema thoroughly — 48 entries,
each with a default, type, consumers and lifecycle, plus constructor validation. That
is contract coverage, not behavioural coverage, and the distinction is the whole
finding: a key can be fully specified and never once change a simulation in a test.

Against the constitution registry the picture is better than it first looked.
Matching by class name flagged `OrthoPotential` as untested; it is not, it is reached
through `AffineBodyRod` and `AffineBodyShell` in five cases. `AffineBodyExternalForce`
likewise appears as `AffineBodyExternalBodyForce`. Two registered UIDs have neither a
public header nor a test, `AffineBodyDrivingSphericalJoint` (27) and
`AffineBodyD6Joint` (28), which matches their documented internal status. One real gap
remained: `StrainLimitingBaraffWitkinShell` (819) had two public headers and a
construction test, and no simulation.

## Added

**`94_strain_limiting_baraff_witkin_shell`** hangs a strip from two corners and runs
the same mesh and load under the strain-limiting shell and under `NeoHookeanShell`,
asserting the limited one does not stretch further. The first version asserted nothing:
at cloth-like stiffness neither model stretches — both reach 1.0002 — so the comparison
was vacuous. Softened to 2 Pa and 5,000 kg/m³ the separation is large and the property
is visible: **6.77 against 35.82** maximum edge stretch. The test now also requires the
unlimited model to exceed 2.0, so it cannot silently become vacuous again.

**`95_linear_solver_parity`** runs one hanging Neo-Hookean bar under `fused_pcg` and
`linear_pcg`, the second of which nothing had ever selected. The cross-solver gap is
compared against the same solver's own run-to-run gap, as this directory's
interpretation rules require of any atomic-order comparison. Measured: the run-to-run
gap is **exactly zero** and the cross-solver gap is **3.8e-15 m**, so both are asserted
below 1e-12. The zero is worth recording on its own — the run-to-run irreproducibility
measured elsewhere in this project is contact-related, not inherent to the linear solve
on a small contact-free system.

**`96_contact_d_hat_relative`** pins both halves of the documented override without
depending on how the backend defines its scene diagonal: with the relative gap off, two
scenes differing only in `contact/d_hat` must differ (**0.0157 m**), and with it on they
must not (**0 m exactly**). The second half alone would pass on any scene where d_hat
does not matter; the first is what stops that.

A note for whoever writes the next one: an affine body's motion lives in its transform
and its vertex positions never move, so a position comparison reads the same rest mesh
in every run. `96` was written with `AffineBodyConstitution` first and reported a
difference of exactly zero for two clearly different `d_hat` values.

## Found defective: `uipc_test_backend_cuda`'s `lbvh` case

`apps/tests/backends/cuda/lbvh.cu:614` compares the GPU point query against a
brute-force reference, and **fails on about half of all runs** — 5 of 8 on the
`bunny0.msh` section alone, 3 of 6 on the whole target. It has presumably been doing so
for some time; a single green `ctest` hides it.

The instability is in the test's own reference, not in the query. Across runs the GPU
result is stable at 101,802 pairs while `brute_froce_query_point` returns **0** in the
runs that pass and **94,728** in the runs that fail.

Zero is impossible. The reference tests whether each AABB's centre lies inside each
AABB, and every box contains its own centre, so a correct result has at least
`num_aabb` = 15,788 pairs. **The runs that pass are the broken ones**: an empty
reference makes `set_difference(gt, test)` trivially empty and
`CHECK(diff.empty())` vacuous. The same signature appears on a twelve-AABB mesh
earlier in the target, where the reference also returns zero. So this check has never
meaningfully verified the point-query path — it has alternated between passing
vacuously and failing.

Not established here: why the reference collapses. An empty `AlignedBox` has
`min = +inf`, `max = -inf` and therefore a NaN centre, which `contains` rejects, so
degenerate boxes are the first thing to look at; but that does not by itself explain
why the outcome changes between runs of the same binary on the same mesh. Fixing it is
separate work and should not be folded into a coverage commit.

## Reproduce

```bash
ctest --test-dir build_raw --build-config Release --output-on-failure
for i in $(seq 1 8); do
  ./build_raw/Release/bin/uipc_test_backend_cuda lbvh -c bunny0.msh \
    2>&1 | grep -E "ground_truth.size|All tests passed|failed"
done
```
