# IAQL direct-picker cloth benchmark, 2026-09-14

## Scope and implementation

The owner requested execution of [ADR 0008](../adr/0008-ipc-adjoint-q-learning.md).
Starting revision: `ca3fdc15`, branch `research/ipc-adjoint-q-learning`.
The initial implementation runs from a modified working tree. It adds:

- `iaql_env`: a named native direct-picker variant of `cloth_drag`, reusing
  the 400-vertex cloth builder, task goal sampler and grasp selection. No robot,
  IK, camera or finger contact; 3 translation actions, five BDF1 substeps,
  smooth progress/action-cost reward, full cloth position/velocity and tool/goal.
- `tangent_pass(return_frames=True)`: preserves all substep responses so the
  final velocity derivative includes the penultimate position. Matrices are
  retained projected Newton systems; the history chain contains inertia only.
- `iaql.soft_targets`: refreshes paired scalar/gradient labels through the
  same stochastic squashed-Gaussian SAC backup, including entropy, policy input
  derivatives, masks and target clipping. Physical action tangents are reusable;
  old value-dependent gradient labels are not reused for changing teachers.
- `SACAgent.update_state_batch`: opt-in derivative loss for both Q heads,
  sharing the original actor/temperature/target schedule. Default
  `adjoint_weight=0` leaves existing point-cloud SAC behavior unchanged.
  `state_activation=silu` is used for both benchmark arms.
- `iaql_benchmark`: finite-difference gate, episode-disjoint frozen-teacher
  regression, then an online SAC/IAQL pair. It saves reports and checkpoints.
  Added after the first run (the Claude session that took the branch over):
  `--phase refit` re-fits critics on the saved fixed dataset without a
  simulator (weight sweep, the design's shuffled-label control, the constant
  train-mean slope); the online loop keeps every TD transition and bounds only
  the mechanics sidecar (`--tangent-rows`, rows without a live tangent learn
  values only, `adjoint_valid_fraction` reports the covered share); periodic
  evaluation from a snapshot (`--eval-every`, `--eval-episodes`);
  `--updates-per-step`; `--phase online --online-weights w` runs one arm per
  process so a pair can run in parallel.

- Backend export modes (takeover, after the tolerance probe):
  `LinearSystemAdjointFeature.set_export_mode('last_iterate' | 'converged' |
  'converged_raw')`. After the Newton loop the engine re-detects contact pairs
  and re-assembles the system at the accepted state, projected or with the
  device-side projection switch off (`utils/make_spd.h`, honoured by every
  `make_spd` site and the friction helper's 2x2 projection; not by
  StableNeoHookean-3D's analytic projection); the forward solve is untouched
  and the switch is restored by an RAII guard. `solve()` refuses the raw mode.
  `iaql_env` takes `export_mode`; the probe captures each decision once per
  `--export-modes` entry and reports every mode against the same differences;
  `--guided-steps-min 1` keeps every snapshot one advance past its recover.
  Test: `uipc_test_diff_sim` "linear_system_adjoint_export_modes" (a shell
  triangle at rest: projected and raw agree; compressed to 60 %: they differ,
  same sparsity, projected curvature never below raw).
- Friction's lagged block (after the export-mode smoke): the re-assembly at
  the accepted state also computes `dG_f/dx_prev` per half-plane friction
  pair (central differences of the friction gradient in the previous
  position, step 1e-4 of the smoothing displacement `eps_v*dt`), exported by
  `LinearSystemAdjointFeature.export_prev_coupling()`; `tangent_pass` takes
  it as `prev_coupling` and adds `-B_fric X_{k-1}` to each substep's
  right-hand side beside the inertia term (unit test against the explicit
  recurrence); `iaql_env(friction_chain=True)` and the probe's
  `--friction-chain` switch it on in the converged export modes. Simplex
  (cloth-body) friction coupling is not exported yet.

The first run's replay stored all transition/tangent fields in one bounded row
of 1,024 and evicted them together; its `report.json` arguments lack the
sidecar keys. Neither is the large-scale sequence-replay sidecar proposed in
the ADR. Trust weights provide hard validity only; no calibrated
contact-conditioned weighting is implemented.

## Machine and command

Linux, NVIDIA RTX PRO 6000 Blackwell, driver 595.84, CUDA driver capability 13.2,
existing Release backend with CPython 3.13 bindings, PyTorch 2.12.0+cu130 from
the Genesis interpreter. Other training processes were already using the GPU
(about 51 GiB allocated and 100% utilization); they were left running. These
shared-device wall times are diagnostic and do not establish a speedup.
The benchmark uses native libuipc without initializing Genesis itself.

```bash
env PYTHONPATH=build/python/src:python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/ge47gax/kun/genesis-world/.venv/bin/python -m uipc_manip.iaql_benchmark \
  --out output/iaql/silu_friction0_s0 --phase all
```

Defaults: friction 0, 40 settle frames, 150-decision horizon, 10 ms timestep,
6 mm/action unit, Newton velocity tolerance 0.001, four probe states, normalized
FD steps 0.03 and 0.1 with two repeats, 128 collected transitions, 1,000 frozen
critic updates, 256 online transitions per arm, two 50-step evaluation episodes.
This is a correctness/smoke budget, not a trained-policy comparison.

## Diagnoses before learning

The first adapter read the ordinary geometry's `velocity` attribute, which
remained at its initial value after `World.retrieve()`. This made the derivative
of the declared state inconsistent with its observed value. The adapter now uses
`FiniteElementStateAccessorFeature.copy_to` with a velocity-bearing state
geometry. Captured steps check the accessor against `(x_new-x_previous)/dt`,
and snapshot restore checks the entire x/v/tool/goal state.

After that correction, the ReLU teacher still failed the joint value-gradient
gate at three of four states. At the first state, tightening Newton tolerance
from 0.001 to 0.00001 gave position tangent errors about 0.12–0.55% and velocity
errors about 0.22–1.90%, but a roughly 39% Bellman-gradient error. An independent
check using the *linearized state map* reproduced the discrepancy: the x slope
was 0.1497 analytically, 0.1496 at FD step 0.0001, but 0.2500 at step 0.03.
The network crossed ReLU activation boundaries; this was not evidence of a
39% mechanics error. The first learning comparison therefore uses SiLU in both
SAC and IAQL state networks and repeats the physics gate. It does not alter the
existing point-cloud networks. The strict-tolerance run also emitted line-search
limit warnings near numerical energy resolution; stricter tolerance alone was
not adopted as a cure.

Artifacts: `output/iaql/friction0_s0.log` (aborted stale-velocity adapter),
`output/iaql/friction0_s0_v2/report.json` (corrected accessor, ReLU gate),
`output/iaql/strict_friction0/report.json` (field-level checks), and
`output/iaql/silu_friction0_s0/` (smooth-network experiment).

## Validation and experiment status

The focused CPU suites check the full soft target against explicit autograd and
finite differences, terminal masks and clipping, detached teachers, mixed
parameter/action derivatives, invalid/zero labels, state checkpoint reload,
baseline equivalence at zero derivative weight, and legacy adjoint/SAC behavior;
the takeover added the sidecar batch (rows without mechanics are valid 0 with
zero tangents) and the refit phase on a synthetic dataset. Full `uipc_manip`
CPU suite after the takeover: 376 passed, 14 CUDA-marked deselected (the seven
`test_iaql` cases included). Live native simulation, tangent export and exact
snapshot restoration have run. No dressing, robust policy gain, exact friction
adjoint, or algorithmic novelty claim follows from these checks.

## Results of the smooth-network run (`silu_friction0_s0`, 1,565 s on the shared GPU)

Gate criterion as coded: a snapshot is accepted only if *every* check (both
finite-difference steps) has Bellman-gradient cosine ≥ 0.95 and relative error
≤ 0.2; the run continues past the gate when at least 75 % of snapshots pass.
This is stricter in form than the ADR's median/tenth-percentile gate and was
applied to four snapshots, not the hundred the ADR asks for. Snapshot `i` is
the settled cloth after a fresh reset followed by `4·(i mod 4)` guided steps
(goal direction plus noise), then a uniform random centre action in
`[-0.5, 0.5]^3`.

| Snapshot | Guided steps | Bellman cosine (ε 0.03 / 0.1) | Bellman rel. error | Reward-gradient rel. error | Position tangent rel. error per axis | Velocity tangent rel. error per axis | Repeat std of the FD gradient |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 1.0000 / 1.0000 | 0.011 / 0.013 | 0.005 / 0.005 | 0.002, 0.003, 0.008 | 0.002, 0.009, 0.020 | ≤ 6e-5 |
| 1 | 4 | 0.9998 / 0.9998 | 0.029 / 0.029 | 0.031 / 0.032 | 0.059, 0.034, 0.034 | 0.036, 0.034, 0.059 | ≤ 1e-6 |
| 2 | 8 | 0.9987 / 0.9987 | 0.083 / 0.083 | 0.078 / 0.077 | 0.135, 0.139, 0.047 | 0.148, 0.147, 0.123 | ≤ 5e-7 |
| 3 | 12 | 0.9990 / 0.9989 | 0.181 / 0.190 | 0.100 / 0.102 | 0.214, 0.139, 0.051 | 0.246, 0.161, 0.122 | ≤ 3e-7 |

All four accepted (4/4); snapshot 3 sits just inside the 0.2 error bound. The
tool rows of the tangent are exact (1e-14). The error is the same at both step
sizes and the repeat scatter is 1e-5 or less, so it is not finite-difference
noise: it is a systematic error of the linearised decision map that grows with
the number of guided steps, i.e. with how far the cloth has been dragged and
bunched against the frictionless table. Direction survives it (cosine ≥ 0.9987
throughout) because the critic's state gradient projects the state error onto
a few directions. Timing per captured decision (five substeps, export and one
host LU factorisation per substep, GPU shared with two training jobs at 100 %
utilisation): forward 1.43–1.86 s, tangent pass 2.3–2.8 ms.

Fixed-teacher critic regression (128 transitions from 8 episodes, 96 train /
32 held-out rows from 2 disjoint episodes, 1,000 updates, gradient scale 0.351
= RMS of the training slopes, both Q heads reported):

| Derivative weight β | Held-out value MSE | Held-out slope MSE | Held-out slope cosine |
|---|---|---|---|
| 0 | 0.397 / 0.366 | 0.126 / 0.127 | 0.384 / 0.352 |
| 0.1 | 0.418 / 0.330 | 0.056 / 0.058 | 0.845 / 0.836 |

Online smoke, 256 steps per arm (64 random warm-up), batch 32, one update per
step, two 50-step evaluation episodes at the end: no success in either arm,
final distances 0.104 / 0.084 (SAC) and 0.107 / 0.081 (IAQL), returns −5.5 /
+6.9 and −6.0 / +7.5; training 423 s and 440 s (1.65 and 1.72 s per step). At
this budget the smoke says only that both arms run and save; it is not a
learning comparison.

## Refit on the saved dataset: is the slope gain the physics? (CPU, no simulator)

`--phase refit` on `silu_friction0_s0/fixed_dataset.npz`: the same
episode-disjoint split, same seeds, 1,000 updates, paired labels versus the
training slopes permuted across rows (held-out labels intact) and the constant
train-mean slope as the trivial predictor.

| Labels | β | Held-out value MSE | Held-out slope MSE | Held-out slope cosine |
|---|---|---|---|---|
| train-mean slope | — | — | 0.166 | −0.004 |
| paired | 0 | 0.397 / 0.366 | 0.126 / 0.127 | 0.384 / 0.352 |
| paired | 0.01 | 0.371 / 0.353 | 0.112 / 0.114 | 0.615 / 0.505 |
| paired | 0.1 | 0.418 / 0.330 | 0.056 / 0.058 | 0.845 / 0.836 |
| paired | 1 | 0.212 / 0.324 | 0.054 / 0.059 | 0.820 / 0.842 |
| shuffled | 0.01 | 0.421 / 0.392 | 0.125 / 0.126 | 0.414 / 0.355 |
| shuffled | 0.1 | 0.372 / 0.413 | 0.129 / 0.125 | 0.287 / 0.372 |
| shuffled | 1 | 1.170 / 0.704 | 0.200 / 0.192 | 0.086 / 0.005 |

Reading: a constant slope explains nothing (the goals differ between
episodes, so the training-mean slope has zero cosine with held-out slopes);
shuffled labels never raise the held-out cosine and at β = 1 destroy both
slopes and values; paired labels raise it monotonically to 0.84 at β ≥ 0.1
with no loss in value error. On this dataset the derivative loss transfers the
physical slope information across episodes and goals. It is a 32-row held-out
set from one seed; the ADR's held-out-slope criterion for learning runs still
needs the larger dataset.

## Newton tolerance: the tangent error is not convergence (`silu_tol1e-5_probe`, 417 s)

The same four states and centre actions probed with `--velocity-tolerance 1e-5`
(the run's default is 1e-3), everything else equal:

| Snapshot | Bellman rel. error (1e-3 → 1e-5) | Position tangent rel. error per axis (1e-5) | Velocity tangent rel. error per axis (1e-5) | Forward per captured decision |
|---|---|---|---|---|
| 0 | 0.011 → 0.006 | 0.001, 0.001, 0.005 | 0.002, 0.003, 0.008 | 2.28 s |
| 1 | 0.029 → 0.029 | 0.058, 0.034, 0.033 | 0.035, 0.033, 0.058 | 3.05 s |
| 2 | 0.083 → 0.082 | 0.136, 0.138, 0.047 | 0.148, 0.147, 0.122 | 3.58 s |
| 3 | 0.181 → 0.189 | 0.215, 0.141, 0.053 | 0.246, 0.162, 0.126 | 4.06 s |

Only the lightly loaded first state improves; at the three dragged states the
errors are the same to three digits at both tolerances and both difference
steps, while the forward cost rises 1.6–2.2× (line-search limit warnings at
energies of 3e-8). The analytic Bellman slope is consistently 10–20 % larger
in magnitude than the difference at those states (e.g. 0.847 vs 0.781 on the
dominant axis of snapshot 2) with cosine ≥ 0.9987. Since snapshot 0 is exact
to 0.1 %, the BDF1 inertia chain and the constraint coupling are right; what
grows with the dragged, bunched cloth is the retained matrix's projection to
SPD of the barrier and shell Hessians, which the ADR names as the approximate
mode's error. Tighter Newton tolerance is therefore not a cure and was not
adopted; the learning runs keep 1e-3. Reducing this error needs the
unprojected Hessian at the accepted state (an exact-mode export), or a trust
weight that discounts these states; both remain open.

## Friction 0.6: the approximate mode fails the gate (`silu_friction0.6_probe`, 222 s)

Same protocol with `--friction 0.6` (friction enabled in the contact model,
default tolerance 1e-3). Repeat scatter of the difference gradient is ≤ 7e-7,
so the frictional solve reproduces and the numbers below are systematic.

| Snapshot | Bellman cosine | Bellman rel. error | Reward-gradient cosine / rel. error | Position tangent rel. error per axis | Velocity tangent rel. error per axis |
|---|---|---|---|---|---|
| 0 | 0.9955 | 0.107 | 1.000 / 0.184 | 0.228, 0.179, 0.027 | 0.414, 0.260, 0.067 |
| 1 | 0.9880 | 0.374 | 0.994 / 0.288 | 0.256, 0.146, 0.074 | 0.213, 0.122, 0.093 |
| 2 | 0.8563 | 0.766 | 0.906 / 0.682 | 0.534, 0.464, 0.312 | 1.028, 0.656, 0.314 |
| 3 | 0.9907 | 0.259 | 0.988 / 0.260 | 0.124, 0.289, 0.189 | 0.160, 0.591, 0.313 |

Accepted 1/4; a full run would stop at the gate. The reward-gradient error is
as large as the Bellman error, so this is the mechanics, not the critic: the
analytic slope underestimates the difference by a factor of 1.6 at snapshot 1
and 3 at snapshot 2 (0.099 against 0.324 on the dominant axis). The retained
matrix holds the friction Hessian at the last iterate, but the inertia-only
chain carries none of friction's dependence on the previous substep's
positions and lagged tangent basis, which the ADR lists as required for a
complete derivative. Direction still holds at three of four states (cosine
≥ 0.988). Consequence: no frictional learning run until the lagged friction
terms enter the chain (exact mode) or a calibrated trust weight discounts
these labels; hard validity alone (finite, converged, small residual) would
have accepted every one of them.

## Matched online pair launched (2026-09-14 16:24)

`run_iaql_pair_now.sh` (session scratchpad), after a 70-step smoke of the new
online path (sidecar eviction at 8 rows, mid-run evaluation from a snapshot,
`adjoint_valid_fraction` 0.06 at step 64, exit 0): two processes in parallel
with `--phase online --online-steps 5000 --eval-every 500 --eval-episodes 3
--eval-steps 50 --updates-per-step 2 --beta 0.1 --seed 0`, arms
`--online-weights 0` (`output/iaql/online5k_s0_sac`) and `0.1`
(`output/iaql/online5k_s0_iaql`), friction 0, tolerance 1e-3, a 1,024-row
sidecar over an unbounded TD replay, the same network seed, the same first 64
random actions and reset seeds. At about 1.7 s per step plus ten evaluations
of 150 decisions each arm needs roughly 2.5–3 h on the shared GPU; both wall
times are lower bounds while the other sessions' training jobs run. Results
are not in this record yet.

Two hazards for whoever scales the gate: probe states with `i mod 4 = 0` call
`World.dump()` straight after a `recover()`, the pattern that asserts in the
dressing scene (2026-09-14 physics-gradient entry in the handoff); it has held
here for the settled initial state, but each such snapshot should follow an
advance before the gate grows to the ADR's hundred states. And the coded gate
is per snapshot over all checks, stricter than the ADR's median form; report
which one was applied.

## Export modes: the projection is the whole tangent error (smoke, 2 states)

Three modes of the same restored state and centre action, differences taken
once. `last_iterate -> converged` isolates the evaluation point and the
contact pair set; `converged -> converged_raw` isolates the SPD projection.
Friction 0, tolerance 1e-3, one guided step before snapshot 0 and five before
snapshot 1, `output/iaql/modes_smoke`:

| Mode | Position tangent rel. error per axis (snap 0 / snap 1) | Velocity tangent rel. error (snap 0 / snap 1) | Bellman rel. error (snap 0 / snap 1) | Forward per captured decision |
|---|---|---|---|---|
| last_iterate | 0.016, 0.013, 0.023 / 0.081, 0.049, 0.037 | 0.019, 0.011, 0.051 / 0.059, 0.052, 0.070 | 0.042 / 0.037 | 2.6–3.1 s |
| converged | 0.016, 0.013, 0.023 / 0.081, 0.049, 0.037 | 0.019, 0.011, 0.051 / 0.059, 0.052, 0.070 | 0.041 / 0.037 | 2.8–3.3 s |
| converged_raw | 0.0014, 0.0015, 0.0023 / 0.0003, 0.0006, 0.0010 | 0.0015, 0.0016, 0.0059 / 0.0008, 0.0007, 0.0017 | 0.006 / 0.0003 | 3.1–3.2 s |

Re-assembling at the accepted state changes nothing to three digits, so
neither the evaluation point nor the pair set was the error; switching the
projection off removes it: the raw tangent agrees with the differences to
0.03–0.2 % and the Bellman gradient to 0.03–0.6 %, against 2–8 % and 4 % with
the projected matrix. The re-assembly costs one extra detection and assembly
per frame, about 0.2 s per five-substep decision here on the shared GPU. The
host LU factorises the raw (possibly indefinite) matrix without trouble at
1,200 degrees of freedom. `uipc_test_diff_sim` (3 cases, 563 assertions)
passes on the new build: at rest the projected and raw re-assemblies agree
to 1e-8, compressed to 60 % they differ and the projected curvature is never
below the raw.

## Export modes on 100 states (`modes100_s0`, 6,060 s on the shared GPU)

The ADR's gate size: 100 snapshots, seeds 100–199, one guided step plus
`4·(i mod 4)` (so 1, 5, 9 or 13 guided drag steps before the centre action),
friction 0, tolerance 1e-3, each centre decision captured in all three modes
against one set of differences (ε 0.03; ε 0.1 agrees to three digits, repeat
scatter ≤ 1e-5). Per-axis tangent errors pool the three axes of every state.

| Mode | Position tangent rel. error median / p10 / p90 / max | Velocity tangent rel. error median / p90 / max | Bellman-gradient rel. error median / p90 / max | Bellman cosine median / p10 / min | Reward-gradient rel. error median / p90 | Gate (per state, all checks) | Forward per captured decision (median) |
|---|---|---|---|---|---|---|---|
| last_iterate | 0.043 / 0.013 / 0.109 / 0.239 | 0.061 / 0.130 / 0.285 | 0.033 / 0.083 / 0.151 | 0.99985 / 0.99918 / 0.98856 | 0.029 / 0.068 | 99/100 | 2.69 s |
| converged | 0.043 / 0.013 / 0.108 / 0.239 | 0.061 / 0.130 / 0.285 | 0.033 / 0.083 / 0.150 | 0.99985 / 0.99918 / 0.98868 | 0.029 / 0.068 | 99/100 | 2.89 s |
| converged_raw | 0.0020 / 0.0005 / 0.0065 / 0.023 | 0.0028 / 0.0102 / 0.031 | 0.0017 / 0.0075 / 0.145 | 1.00000 / 0.99999 / 0.99026 | 0.0011 / 0.0064 | 99/100 | 2.86 s |

Position tangent error, median by guided steps 1 / 5 / 9 / 13:

| Mode | 1 | 5 | 9 | 13 |
|---|---|---|---|---|
| last_iterate | 0.016 | 0.042 | 0.064 | 0.081 |
| converged | 0.016 | 0.042 | 0.064 | 0.081 |
| converged_raw | 0.0025 | 0.0016 | 0.0017 | 0.0025 |

Reading. Re-assembling at the accepted state reproduces the last-iterate
numbers to three digits on all 100 states, so the evaluation point and the
contact pair set contribute nothing. Switching the projection off divides
the median position tangent error by 22 (4.3 % to 0.20 %), the p90 by 17 and
the Bellman-gradient error by 19, and the growth with drag disappears: the
raw error is 0.16–0.25 % at every drag depth where the projected one climbs
from 1.6 % to 8.1 %. That is the ADR's approximate-mode error identified: the
solver's PSD curvature is the right matrix for the Newton step and the wrong
one for the implicit derivative, and nothing else in the frictionless
decision map is missing at this tolerance. The projected matrix's remaining
spread (max 24 %) is the same states that bunch the cloth most. The one state
every mode rejects (snapshot 44, one guided step) is rejected by the critic,
not the mechanics: its raw reward-gradient error is 0.7 % and its position
tangent 0.1–0.7 %, while its Bellman-gradient error is 14.5 % with cosine
0.990, i.e. the SiLU critic's state gradient still bends within a 0.03 action
unit there. Cost: the re-assembly adds one detection and assembly per frame,
0.2 s per five-substep decision on the shared GPU (2.69 → 2.86–2.89 s);
the raw factorisation costs the host nothing extra at 1,200 degrees of
freedom (tangent pass 2.4–2.8 ms).

## Friction 0.6 with the raw Hessian and the lagged coupling: the gate passes (`fric06_modes`, `fric06_modes_chain`)

Four states (seeds 100–103, 1/5/9/13 guided steps, one more than the earlier
friction probe, so its numbers are not the same states), friction 0.6,
tolerance 1e-3, differences taken once per run. The first run captures the
three export modes without the lagged coupling, the second with it in the
converged modes (`--friction-chain`; `last_iterate` cannot carry it and
reproduces the first run exactly, which checks the two runs against each
other). 880–1,480 coupling blocks per captured decision (five substeps of
180–300 friction pairs).

| Variant | Position tangent rel. error median / p90 / max | Velocity tangent rel. error median / max | Reward-gradient rel. error median / max | Bellman-gradient rel. error per state | Gate |
|---|---|---|---|---|---|
| last_iterate, no chain | 0.247 / 0.352 / 0.429 | 0.218 / 1.303 | 0.310 / 0.536 | 0.260, 0.385, 0.741, 0.274 | 0/4 |
| converged, no chain | 0.243 / 0.352 / 0.426 | 0.218 / 1.288 | 0.310 / 0.533 | 0.260, 0.385, 0.737, 0.274 | 0/4 |
| converged_raw, no chain | 0.256 / 0.385 / 0.446 | 0.251 / 1.332 | 0.350 / 0.544 | 0.251, 0.310, 0.676, 0.309 | 0/4 |
| converged + chain | 0.085 / 0.194 / 0.228 | 0.172 / 0.352 | 0.035 / 0.110 | 0.029, 0.063, 0.038, 0.265 | 3/4 |
| converged_raw + chain | 0.0014 / 0.0044 / 0.0058 | 0.0033 / 0.022 | 0.0011 / 0.0018 | 0.001, 0.001, 0.000, 0.191 | 3/4 |

Reading. With friction the projection alone repairs nothing (raw without the
chain is as wrong as the rest) and the chain alone repairs most of the
mechanics but not all (8.5 % median with the projected friction Hessian):
both are needed, and together they bring the frictional tangent to the
frictionless raw level, 0.14 % median position error and reward-gradient
error at most 0.18 % on every state. The state whose three axes were all
wrong in the first friction probe (the 9-guided-step state, 0.53 / 0.46 /
0.31 position error there and 0.39 / 0.45 / 0.28 here without the chain) is
0.1 % with the chain: it was the missing lagged block, not a stick-slip
switch. The one rejected state (13 guided steps) is again the critic's:
reward-gradient error 0.1 %, position tangent 0.1–0.6 %, Bellman error 19 %
with cosine 0.993. The lagged block costs nothing measurable (forward per
captured decision 2.09–2.33 s either way; the coupling kernel is six
gradient evaluations per pair). Consequence: the frictional regime is now a
mechanics-validated regime for derivative labels on the half-plane contact;
what remains untested is the simplex (cloth-body) friction, whose coupling
is not exported, and states that actually switch stick/slip within a
difference step, of which these four contain none.

## The matched 5,000-step pair: inconclusive, the benchmark is not yet learnable at this budget

`online5k_s0_sac` and `online5k_s0_iaql` (two processes in parallel, friction
0, tolerance 1e-3, `last_iterate` export, β 0.1, two updates per step, batch
32, 64 random warm-up steps, unbounded TD replay, 1,024-row mechanics
sidecar, same seeds; three 50-step evaluation episodes every 500 steps).
Pure training time 13,135 s (SAC) and 13,288 s (IAQL), 2.63 and 2.66 s per
step with both arms, the 100-state probe and two other sessions' jobs on the
GPU. At the end the sidecar covered 22 % of the sampled rows
(`adjoint_valid_fraction` 0.219), derivative loss 0.077, critic loss 0.36
(IAQL) against 0.54 (SAC).

| Steps | SAC mean return | SAC mean distance | SAC successes | IAQL mean return | IAQL mean distance | IAQL successes |
|---|---|---|---|---|---|---|
| 500 | −0.41 | 0.096 | 0 | −0.24 | 0.095 | 0 |
| 1000 | 0.21 | 0.093 | 0 | 0.22 | 0.092 | 0 |
| 1500 | 0.17 | 0.093 | 0 | 0.25 | 0.092 | 0 |
| 2000 | 0.19 | 0.093 | 0 | 0.05 | 0.093 | 0 |
| 2500 | 0.12 | 0.093 | 0 | 0.43 | 0.091 | 0 |
| 3000 | 0.30 | 0.092 | 0 | 0.17 | 0.092 | 0 |
| 3500 | 0.62 | 0.090 | 0 | 0.64 | 0.089 | 0 |
| 4000 | −2.70 | 0.109 | 1 | −1.69 | 0.103 | 0 |
| 4500 | −1.91 | 0.105 | 0 | −0.21 | 0.094 | 0 |
| 5000 | 2.54 | 0.078 | 0 | 2.62 | 0.077 | 0 |

Reading. For 3,500 steps both policies barely move the marker (per-episode
returns within ±5, distances at the start values); from 4,000 steps both
start taking large actions with returns of ±10 in single episodes, one SAC
success at 4,000, and the last evaluation of both arms is the best so far
(mean distance 0.078 / 0.077) with no success. The two arms are
indistinguishable at every point against a three-episode evaluation whose
per-episode spread is ±5. This is the outcome the design anticipated for an
underpowered budget: it is not evidence for or against the derivative loss.
Before any SAC/IAQL comparison the benchmark has to become learnable, which
means a per-decision cost far below 2.6 s (a vectorised world with several
cloths, or a smaller cloth) and a budget at which vanilla SAC shows a curve;
only then do paired seeds and the shuffled-label online control mean
anything. Checkpoints: `output/iaql/online5k_s0_{sac,iaql}/online_beta_*.pt`.

## Actor-gradient fidelity: the exact label beats both critics' action gradients (`fidelity_s0`, 2026-09-15)

The ADR's counterfactual-action check on the two 5,000-step checkpoints, with
the GPU otherwise idle (0.12 s per captured decision, the whole probe 2 min).
Ten states of the frozen SAC policy (1–13 policy steps after a reset, seeds
300–309), centre action the policy's mean clipped to ±0.9. Reference: the
finite-difference gradient of the 8-decision return under the frozen policy
(first action ±0.1 per axis, then the policy, terminal value from the
policy's own target critic). Every candidate is rolled out along ±0.1·unit(g)
and its own return change is measured.

| Direction | Mean cosine with the return gradient | Return improved (states) | Mean return improvement | Mean distance gain (m) |
|---|---|---|---|---|
| reward gradient only, `dR/du` through the exact tangent | 0.857 | 10/10 | +0.151 | +0.00105 |
| exact one-decision label, SAC critic's continuation (`ipc_sac`) | 0.842 | 10/10 | +0.149 | +0.00089 |
| exact one-decision label, Sobolev critic's continuation (`ipc_sobolev`) | 0.852 | 10/10 | +0.145 | +0.00096 |
| SAC critic's own `dQ/da` (`dq_sac`) | 0.325 | 8/10 | +0.038 | +0.00042 |
| Sobolev critic's own `dQ/da` (`dq_sobolev`) | 0.604 | 9/10 | +0.087 | +0.00064 |

Per state the SAC critic's action gradient points the wrong way twice
(cosine −0.95 and −0.28, return −0.254 and −0.076) and the Sobolev critic's
once (−0.56, −0.151); the exact label never does. Readings. (1) Used directly,
the exact one-decision label improves the frozen policy's 8-step return at
every state and four times as much as the SAC critic's own action gradient,
which is the actor-side line's claim in the benchmark's units. (2) The
critic-side transfer works: the Sobolev critic's `dQ/da` is twice as aligned
and twice as effective as the SAC critic's at the policy's own action, but it
still delivers only 60 % of the direct label's improvement, the cost of the
second approximation layer. (3) At these states the reward gradient alone is
as good as the full label: the continuation term `γ Dᵀ∇V̄(s')` adds nothing
on average with 5k-step critics, and at one state the SAC critic's
continuation destroys the direction (`ipc_sac` 0.09 where `ipc_sobolev` is
0.79), so the value gradient is the weak factor, as the rejected gate states
already said. The ranking direct label > Sobolev critic > SAC critic is the
one the combined learner's arms test online next.

## Five matched arms at 20k transitions, 64 lockstep slots (`vec20k_s0_*`, 2026-09-15)

The first round on the throughput rebuild (64 cloths per World, batched GPU
tangent, GPU learner). All arms: seed 0, 64 slots, 20k transitions (313
lockstep steps, one random warm-up step), two updates of batch 32 per
transition, `converged_raw` label, continuation trust κ = 2, Gaussian action
locality σ = 0.5, evaluation of 64 deterministic 50-step episodes every 2,000
transitions and at the end. The mechanics arms ran four in parallel on the
GPU tangent (109–116 ms per transition each); the SAC arm ran alongside
five processes (68 ms). An 8-slot round earlier the same day (SAC 12.85 /
0.024 m / 3 of 8; the κ = 0 hard-gate mix arm 11.80 / 0.030 / 1 of 8) learns
faster per transition than the 64-slot round (one random warm-up step, 64
correlated streams), so arms are compared only within a round.

| Arm | Actor term | Critic slope loss | 4k | 8k | 12k | 16k | 20k: mean return / mean distance / successes |
|---|---|---|---|---|---|---|---|
| SAC64 | – | – | 2.72 / 0.086 / 1 | 5.80 / 0.068 / 4 | 6.45 / 0.064 / 4 | 7.29 / 0.059 / 7 | **6.78 / 0.062 / 9 of 64** |
| critic_trust | – | β 0.1 | 3.37 / 0.082 / 1 | 6.13 / 0.066 / 4 | 6.35 / 0.065 / 8 | 6.72 / 0.062 / 10 | **6.58 / 0.063 / 5 of 64** |
| actor_trust | mix ρ 0.5 | – | 4.88 / 0.073 / 6 | 8.53 / 0.051 / 14 | 11.15 / 0.035 / 14 | 13.19 / 0.023 / 33 | **13.31 / 0.022 / 29 of 64** |
| both_trust | mix ρ 0.5 | β 0.1 | 9.22 / 0.047 / 5 | 9.51 / 0.045 / 3 | 9.90 / 0.043 / 3 | 11.62 / 0.033 / 15 | **12.33 / 0.028 / 18 of 64** |
| shuffled_trust | mix ρ 0.5, labels permuted | β 0.1, labels permuted | −5.38 / 0.135 / 3 | 7.22 / 0.059 / 9 | 1.62 / 0.092 / 10 | 8.65 / 0.051 / 5 | **9.05 / 0.048 / 9 of 64** |

Final learner statistics: the actor arm's critic loss 0.14 and Q mean 11.6
against SAC's 0.46 and 6.0; its label's continuation-to-reward ratio 1.05
with continuation trust 0.74 and twin disagreement 0.39 (SAC-side critics
never learned enough for the trust to rise: the critic arm's ratio 0.35);
the sidecar covered 6 % of the sampled rows (1,024 rows over 16k), so the
critic slope loss acted on few rows and its `adjoint_loss` was 0.000–0.004.

Reading. (1) The direct actor use of the exact one-decision label doubles
the return over SAC (13.3 against 6.8), cuts the final distance from
0.062 m to 0.022 m and lifts the success count from 9 to 29 of 64, monotone
from 4k transitions on; this is the actor-side line's claim in a benchmark
with a healthy baseline, one seed. (2) The critic slope loss adds nothing on
its own at this sidecar coverage and slightly reduces the actor arm's gain
when combined (12.3 against 13.3): direct label > Sobolev critic, as the
fidelity probe predicted, and the Sobolev term is not the core. (3) The
shuffled control lands between SAC and the actor arm (9.05): a third of the
actor-side gain does not need the label's semantics (a permuted label still
replaces part of the critic's gradient by a reward-shaped direction and adds
exploration noise), two thirds do. An actor-only shuffled control and more
seeds are the next round. (4) Both mix arms' Q means are twice SAC's with
lower critic loss: the policy improvement feeds back into the value.

