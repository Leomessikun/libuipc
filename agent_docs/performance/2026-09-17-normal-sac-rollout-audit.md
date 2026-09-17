# Ordinary SAC dressing: rollout and training audit

Status: completed analysis of saved SAC results and two bounded, unchanged-policy
geometry captures. No training, policy update, solver change or new algorithm.
The owner abandoned IPC policy-gradient research and then clarified twice that
the immediate subject is **ordinary SAC policy rollouts**, not BC or IPC-guided
policies. New research proposals are deferred until this diagnosis is established.

## Scope and checkpoint identity

The main checkpoint is
`output/uipc_manip/dressing_redesign_20260916/warm_sac/checkpoints/checkpoint_00127416.pt`.
It is ordinary SAC: physics actor and adjoint weights are zero, no imitation
guidance, dense action-conditioned critic, plain trunk, single-frame Wang-flow
actor with PointNet2. It inherited 125,016 transitions and Adam/replay from
`abl_dense_s1`, then collected 2,400 additional transitions on `tshirt_26`.
It is not an actor-only BC checkpoint and is not the best historical SAC model
across all architectures. The source trained on five garments and 45 body poses
in region 13. The continuation sampled eight training poses of one garment.

Saved evidence reanalysed:

- `expert_pretrain_20260917/evaluation.json`: only its `sac` rows, eight full
  episodes on four configurations, two rounds. The directory name does not make
  this checkpoint a BC policy.
- `dressing_first_principles_20260917/evaluation.json`: only `source` rows,
  four full episodes of the unmodified 125,016-transition SAC checkpoint.
- `abl_dense_s1`, `abl_both_s1`, `abl_both_s2`: training/evaluation CSVs,
  configurations and checkpoint metadata. Exclude post-125,016 continuation
  entries from the architecture learning-curve figure.
- `dressing_redesign_20260916/policy_repeats.json`: repeated final SAC evaluations;
  the IPC arm is outside this diagnosis.

## What the ordinary SAC rollouts actually do

The saved 127,416-transition checkpoint reaches some upper-arm progress in all
eight inspected episodes, but finishes successfully in none. Two distinct
patterns recur; a universal failure to insert onto the forearm is not supported.

| Garment / body | Final upper-arm progress, rounds 1 / 2 | Observed pattern |
|---|---:|---|
| tshirt_26 / 14046 | .513 / .560 | Advances above the elbow, then stalls below completion |
| tshirt_26 / 14047 | .658 / .666 | Advances further but still below the .7 threshold |
| tshirt_68 / 14046 | 0 / 0 | Peaks at .246 / .274, then loses the upper-arm intersection |
| tshirt_68 / 14048 | 0 / 0 | Peaks at .233 / .212, then loses the upper-arm intersection |

All eight episodes have **zero collision and tether command rejections**. The
last 50 decisions average .48–2.12 mm of accepted anchor translation per decision.
These failures are not explained by a controller refusing all requested motion.
Accepted anchor movement is not evidence of useful cloth movement.

There are 109 decisions exceeding the 2 cm grasp-tracking limit out of 2,400.
However, both `tshirt_68/14048` episodes have valid grasp throughout and still
fail. Grasp compliance is a problem in some episodes, not a universal explanation.

## Geometry capture: separating loss of progress from a metric-only story

One new deterministic-policy evaluation used the same four configurations,
seed block 1097–1100, full 300-decision horizon and unchanged saved environment
contract. It records the complete cloth mesh, opening vertices, arm mesh,
landmarks, observations, actions and commanded gripper positions. No snapshot
branch optimization or learning was performed. Fresh execution is not bitwise
identical to an old rollout, so its results are reported separately.

On `tshirt_68/14046`, the opening centroid's perpendicular distance from the
upper-arm axis grows from **25.5 mm at decision 120 to 226.9 mm at decision 180**.
The rendered geometry shows the opening displaced away from the upper arm.
The policy subsequently keeps that failed configuration rather than returning
the sleeve to a useful alignment. On `tshirt_68/14048`, the same qualitative
displacement occurs with **no grasp-limit violation throughout the episode**.
This supports a failure of the executed motion strategy, rather than attributing
every zero-progress reading to rejected commands or an invalid grasp.

On `tshirt_26/14046`, progress is .549 at decision 120 and .569 at 300. The last
50 decisions move the anchor along a 38.7 mm path but only 14.9 mm net; the opening
centroid moves 9.6 mm net. The policy mostly remains in a partially dressed state.
This observation does not distinguish a critic-induced local optimum, missing
state information, a poor action parameterization or a difficult mechanical state.

The progress metric also has real discontinuities. On that same tshirt_26
rollout, it falls from .371 to zero at decision 77 while the opening centroid
moves only 3.78 mm; progress later recovers. The code tests intersections of arm
rays with a fan triangulation of the opening. Losing an intersection abruptly
changes both progress and the task-reward branch. A zero reading alone cannot
be interpreted as instant removal of the entire sleeve. This is a measurement
limitation, not proof of an implementation error or reward hacking.

Recomputing every new upper-arm metric from the saved geometry agrees with the
logged metric to 1.12e-16. Thus the observed discontinuities are in the geometric
definition, not CSV aggregation. Contact forces and contact histories were not
captured; the recordings do not identify a frictional snag or solver defect.

## Does SAC fail only on unseen bodies?

An additional two-slot capture uses poses present in the source training draw;
`tshirt_26/14038` also appears in the recent continuation's draw. Same checkpoint,
300 decisions and seed block 1097–1098; no learning.

| Configuration | Final / peak upper-arm progress | Decisions above grasp limit | Interpretation |
|---|---:|---:|---|
| tshirt_26 / 14038 | .720 / .725 | 15/300 | Geometric completion; not whole-episode grasp-valid success |
| tshirt_68 / 14018 | .150 / .182 | 95/300 | Fails on a previously sampled training configuration |

Both have zero controller rejections. The first also triggers early-turn twice,
so it does not pass the historical paper filter. These two episodes are a small
diagnostic, not a training-set success estimate. They demonstrate that the
problem cannot be reduced to generalization to unseen bodies alone.

## What the training records establish

1. **The trajectory folder was not used by this ordinary SAC run.** The source
   starts with an empty replay, no representation initialization, no teacher
   labels and no imitation loss. Its replay contains its own observations,
   actions, rewards and successors. The 2,400-transition continuation reloads
   that same SAC replay. This is expected for the baseline, but having many
   expert trajectories elsewhere did not give this run successful supervision.

2. **Successful experience is scarce in the logged source run.** The source's
   rolling training success is between 0 and .025 in recorded rows. Its 125,016
   transitions include 17 completed vector episodes at 24 slots: 408 complete
   trajectories, plus partial trajectories, across a plan with 225 garment/body
   configurations. That is 1.81 completed episodes per configuration on average,
   not 125,000 independent dressing attempts. Actual visitation is uneven;
   these aggregate logs do not establish per-configuration coverage or prove
   that a particular additional data budget would solve learning.

3. **Ordinary SAC's evaluation performance is not stable across checkpoints or
   seeds.** Dense/plain seed 1 peaks at 1/25 reported geometric successes and
   finishes at 0/25. Dense/residual seed 1 reaches 7/25 at 110,256 transitions,
   then records 0/25 at 125,016; dense/residual seed 2 records no successes in
   its logged evaluations through 125,016. The 7/25 is a historical snapshot,
   not a robust repeated result or a grasp-validated result. The latest dense
   checkpoint is therefore not sufficient to characterize the strongest
   ordinary SAC result. Changing evaluation seed blocks and native trajectory
   variability prevent attributing the entire drop to parameter updates alone.

4. **The objective rewards partial progress repeatedly.** The implementation
   pays an absolute dressing-progress reward each step; completion is a reported
   threshold, not an absorbing successful terminal state. Thus parking at
   partial progress earns reward. That is compatible with observed stalls,
   but does not prove a reward local optimum: no matched continuation test has
   established the value of a successful alternative from these exact states.
   Critic loss, gradient norm and entropy logs do not isolate the cause.

5. **Most elapsed work is outside network updates.** In the source checkpoint's
   cumulative timers: environment 17,813.7 s, evaluation 10,164.1 s, updates
   4,078.3 s, plus rebuild/other costs. Updates are approximately 12.6% of these
   recorded components. In the recent 391 s SAC training loop, simulation is
   312.5 s and updates 77.6 s. Faster neural updates alone cannot remove the
   principal wall-time cost, and high GPU utilization would not establish useful
   learning. `updates_per_step=0` selects the automatic update budget; it does
   not mean SAC performed zero updates.

## Diagnosis and decision

Confirmed: ordinary SAC learns insertion and some upper-arm advancement, but its
motion around and beyond the elbow is garment-dependent and unreliable. It can
pull the opening away from the arm or settle into partial completion. This also
occurs on a sampled training configuration. Some trajectories exceed the grasp
limit, while a separate fully grasp-valid failure rules that out as the sole
cause. Successful training experience is scarce, and historical checkpoint/seed
results fluctuate substantially.

Not yet established: whether the primary learning cause is critic action ranking,
insufficient successful state coverage, observation ambiguity, exploration,
reward structure, or a specific solver/contact discrepancy. The current data
do not justify claiming that a new RL algorithm, force inputs or more memory
will fix it, or that IPC cannot help. BC findings are excluded from this diagnosis.

Before choosing a new research direction, the next useful comparison is between
the saved best and final **ordinary SAC** checkpoints on one fixed configuration
matrix, with identical success definitions and complete geometry traces. Then
test the leading causal explanation at an identified SAC failure window. Do not
restart the abandoned IPC correction experiments or respond with another broad
algorithm proposal. Keep training stopped while selecting that specific test.

## Artifacts and verification

Before revision: `236b21e8`, branch `research/ipc-adjoint-q-learning`; tracked tree
clean before this documentation change, pre-existing untracked files preserved.
Runtime: existing `build_raw` Release native CUDA 12.8 build, RTX PRO 6000
Blackwell, driver 595.84, existing `genesis-world/.venv` Python environment.
No build, asset revision, solver tolerance, controller or policy parameter changed.

Artifacts under `output/uipc_manip/policy_results_audit_20260917/`:

- `sac_results.py`, `sac_episodes.json`: SAC-only historical trace aggregation.
- `sac_rollouts.png/pdf`, `sac_training.png/pdf`: full rollout and learning curves.
- `capture_geometry.py`, `sac_geometry/`: four unseen-pose captures, 1,200
  decisions, 147.77 s including setup/serialization.
- `capture_training_cells.py`, `sac_training_cells/`: two sampled training-pose
  captures, 600 decisions, 92.91 s with the same timer scope.
- `inspect_geometry.py`, `geometry_numbers.py`, `geometry_findings.json`,
  `sac_geometry_0.png` through `sac_geometry_3.png`: mesh views and displacements.
- `native_summary.json`, `validate_capture.py`, `capture_validation.json`:
  all 1,800 completed SAC decisions checked for complete ordering, finite geometry,
  reward-geometry agreement and command acceptance; no simulator errors.

The initial broad analysis script also contains BC rows; it is not the basis of
the SAC-only conclusions. An initial BC capture failed during setup because of
an incorrect diagnostic attribute name; the corrected BC attempt was stopped
with SIGTERM when the owner clarified SAC-only scope. Those artifacts remain as
`geometry_failed_setup*` and `bc_geometry_stopped*`, are incomplete, and are not
counted as SAC results. All subsequent SAC captures completed; no jobs remain.

Re-run either capture only into a fresh output path, using
`PYTHONPATH=build_raw/python/src:python`, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`,
`LD_LIBRARY_PATH=build_raw/Release/bin:/home/ge47gax/Toolchain/uipc_cuda128/lib`
and `/home/ge47gax/kun/genesis-world/.venv/bin/python`.
