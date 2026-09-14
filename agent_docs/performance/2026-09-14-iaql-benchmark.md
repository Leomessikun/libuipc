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
