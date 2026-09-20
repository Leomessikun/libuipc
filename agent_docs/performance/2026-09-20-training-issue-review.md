# Dressing training inventory: evidence review and repair order

Date: 2026-09-20. Branch: `research/flow-latent-steering`.
Reviewed source baseline: `b731e921`.

## Decision

The [inventory](2026-09-20-training-issue-inventory.md) identifies real problems,
but several of its explanations and impossibility claims do not follow from the
measurements. Do not use it to close dressing RL, certify a privileged upper
bound, or declare RAL novel. No robust learned dressing result has been established.
Equally, these experiments have not established that IPC makes learning impossible.

Prioritize a common task/evaluation contract and physical control timescale,
then a bounded numerical-accuracy comparison, then a matched policy-improvement
experiment. Do not extend the existing 270k-transition run unchanged.

This review re-read code, run configurations, CSVs and saved branch results,
executed a CPU-only analysis and an actual-reward geometric counterexample, and
checked primary literature. It launched no simulator, training or GPU job. This
distinction matters: the repair experiments below are proposals, not results.

## 1. A missing confound: the privileged run changed the control problem

Actual `config.json["env"]` and `["sac_config"]`, rather than CLI defaults:

| Quantity | Historical `wang_teacher_r13_s1` | `state_ub_clean_s1` |
|---|---:|---:|
| Physics dt | 1/60 s | 1/60 s |
| Physics frames per decision | 6 | 1 |
| Decision period | 0.1 s | 0.01667 s |
| Episode decisions / physical duration | 300 / 30 s | 900 / 15 s |
| Maximum translation per axis per decision | 8.660 mm | 1.443 mm |
| Maximum rotation per enabled axis per decision | 5 degrees | 5 degrees |
| Corresponding angular-rate limit | 50 degrees/s | 300 degrees/s |
| Discount | 0.995 | 0.9983333 |
| Discount e-folding time, `-decision_dt / log(gamma)` | 19.95 s | 9.99 s |
| Reward multiplier in replay | 0.5 | 0.1666667 |
| Training cells | 24 sampled cells | all 25 evaluation cells |
| Recorded training elapsed time | 94,161.5 s | 16,469.5 s |

Both saved environments use friction **0.3**. The CLI argument `friction=0.6`
does not describe their effective dressing configuration. Likewise the CLI
`max_translation=0.006` is not their executed translation limit.

`DressingConfig.max_translation` scales with physical decision duration, whereas
`GenesisIPCDressingEnv.step` scales rotation by the fixed `max_rotation`. Thus
the angular and translational exploration regimes changed differently. The
horizon-based discount helper does not preserve physical discount time here.
An equal transition count is neither equal simulated time nor equal workstation
cost. The state actor is also trained on different cells and uses a different
network. These are useful engineering runs, not a controlled perception ablation.

The state run's 22 evaluations really have zero peak upper-arm coverage. But its
35-dimensional input contains geometric summaries, not complete cloth positions,
velocities, folds or contact state. Calling this a full-information upper bound
is unjustified. At its last evaluation, mean **peak** forearm ratio is 0.419;
mean **final** ratio is 0.305. Peak threading is 0.80, final threading 0.16.

At 12,500 transitions, `init_steps=0` and evaluation calls the deterministic
trained actor. The reported 0.52 is the fraction that threaded **at any time**;
the final fraction is 0.08. It is not a random-action baseline.

Time discretization can shrink action-value differences and change exploration.
This is an established issue, not evidence specific to an IPC failure. Tallec,
Blier and Ollivier analyze it explicitly; their smooth-system theory does not
by itself prove the cause of this cloth-contact result.
[Primary source](https://proceedings.mlr.press/v97/tallec19a/tallec19a.pdf).

## 2. The reward is not mathematically flat at both joins

Equal function values on the two sides of a join do not imply zero slope.
The task term is `-distance`, then forearm distance, then
`forearm_length + 5 * upperarm_distance`.

The accompanying analysis calls the actual `wang_progress` function on a planar
hexagonal cuff advancing along a straight arm: finger x=0, elbow x=1, shoulder
x=2. It uses the same four-triangle construction, with a distant body cloud to
isolate the geometric reward. Selected outputs:

| Opening x (m) | Task reward | Total reward |
|---:|---:|---:|
| -0.002 | -0.002 | -0.002 |
| -0.001 | -0.001 | -0.001 |
| 0.001 | 0.001 | 0.001 |
| 0.002 | 0.002 | 0.002 |
| 0.998 | 0.998 | 0.998 |
| 0.999 | 0.999 | 0.999 |
| 1.001 | 1.005 | 1.025 |
| 1.002 | 1.010 | 1.030 |

The local task slopes are **1, 1 and 5**. The upper-arm alignment bonus can
also make the *total* reward discontinuous. This counterexample refutes A1's
mathematical argument; it does not show that a physically feasible motion through
a jam has a useful slope. Weak control authority, branch geometry, unhelpful
exploration, missing experience and a poor critic remain distinct hypotheses.

The genuine objective issue is the mismatch between instantaneous progress and
successful completion with grasp retention. Progress rewards already favor
holding coverage over losing it, so A4 is too absolute. However, the objective
does not explicitly implement the whole-episode grasp constraint or a verified
completion/hold phase. Those omissions deserve a controlled repair.

## 3. What the noise and branch measurements establish

The quantity in C1 is **best candidate minus runner-up**, divided by pooled
within-candidate standard deviation. It is not the RL advantage relative to the
current policy. For example, candidate returns `[0, 1, 1]` have a zero top-two
gap but permit a gain of 1 if the current policy chooses the first action.
Resolving which of two equally good actions is best is unnecessary for improvement.

Noisy samples can estimate expected returns. A comparison of sample means has
uncertainty that depends on repeat count. For independent repeats its standard
error is approximately `sqrt(s_a^2/n_a + s_b^2/n_b)`, not the pooled single-rollout
SD alone. Paired comparisons can help only with justified coupling; restoring
the same initial geometry does not create identical subsequent randomness.
Adaptive candidate selection also needs selection-bias control or independent
confirmation. Thus 1.31 SD does not imply random-walk learning or impossibility.

Recomputing `ral_calibration_20260920/t26_14046/result.json` gives:

| Hold / perturbation | Median absolute mean difference / pooled SD | Positive mean differences | Positive differences >3 SD | Negative differences <-3 SD |
|---|---:|---:|---:|---:|
| 1 / 1.0 | 1.450 | 11/24 | 2/24 | 2/24 |
| 4 / 0.5 | 2.612 | 12/24 | 5/24 | 5/24 |
| 4 / 1.0 | 5.287 | 18/24 | 16/24 | 2/24 |

This is a useful **local positive signal**. The often-quoted 75% is 18/24
absolute separations, comprising 16 improvements and two deteriorations.
It is not an 18/24 dressing success rate or a calibrated confidence guarantee.

Limits: one garment/body, one displacement direction per state, five repeats,
16-decision continuation, and a base command held for the same prefix duration.
The baseline is not necessarily the ordinary feedback policy over that prefix.
All initial upper-arm ratios are already 0.474–0.592. These states therefore do
not locate the bottleneck of the new state actor, which never reaches the upper
arm. The decision period is 0.1 s: **four decisions means 0.4 s**, not the
0.0667 s of four decisions in the state run.

The calibration used 16,800 decisions in 1,579.4 s, or 10.64 decisions/s including
its measured overhead. At that observed rate, 480 decisions cost about **45 s**,
not 12 s. That extrapolation is still only a rough cost estimate, not a production
throughput benchmark.

The tolerance probe is the most actionable numerical lead, with important limits:

- The 139x positional-divergence and 583x coverage-range reductions are local
  repeatability measurements, not improvements in RL learning speed or success.
- Both tolerances changed together; each setting generated its own approach
  trajectory and snapshot. This is not the same starting physical state across
  settings. There are only three repeats in one cell at one snapshot.
- Range is not standard deviation; a reduction in range cannot be multiplied
  into a different experiment's top-two-margin/SD ratio.
- The reported 2.5x time ratio used a shared GPU. A tighter solve can also change
  mean dynamics and action rankings. Reduced variance alone is not accuracy.
- The nominal absolute Newton displacement stopping threshold is `velocity_tol
  * dt` in `max_translation_checker.cu` (unless scene-relative tolerance is
  enabled): 1.67 mm at 0.1 and dt=1/60. That is comparable to a full per-axis
  translation in the 60 Hz run. This is **not** a position-error bound or proof
  that a command is ignored; it motivates checking action effects against
  solver convergence, rather than treating this threshold as harmless.

The recorded zero restore error checks positions. Full-state recovery needs its
own verification, including velocity, targets, controller/history state, and
relevant solver state. The fixed-command predictability experiment establishes
divergence; some other macro studies use closed-loop commands. They cannot all
be described as identical-command repeats. Earlier evidence even finds contact
stabilizing some phases, so uniform contact amplification is not established.

Force dispersion makes the present readout unsuitable for an **uncalibrated**
force-based objective or safety certificate. It does not prove every force input
or temporal force statistic unusable. Validate absolute errors, tolerance
convergence, units, net versus per-contact quantities, and sensor-like temporal
aggregation before reconsidering it. No arbitrary force threshold is justified.

## 4. Point-by-point disposition

"Observed" below means that the limited recorded observation is supported, not
that its proposed cause has been established.

| Item | Assessment / correction |
|---|---|
| A1 | Incorrect inference; actual reward counterexample above. Contact-conditioned plateaus need separate measurement. |
| A2 | Zero upper-arm coverage observed. A workstation cost *per reward plateau* is not established; control timescale and budget comparisons are confounded. |
| A3 | Supported: no explicit grasp term/constraint matching the whole-episode success gate. |
| A4 | Three of four stop interventions succeed. Holding progress is already rewarded indirectly; there is no explicit verified completion phase. |
| B1 | Inconsistent evaluators are real. "Every study since September 17" is false: current `train_sac.evaluate` still copies final-step `valid_grasp_success` and does not require a 12-step coverage tail. |
| B2 | Two geometries exist. There are **two different 9/25 claims**: a 5 cm-extension evaluation, and historical 6 plus three stopping rescues. The stopping and driving artifacts both have extension **0.0**. |
| B3 | Verified for the historical SAC CSVs inspected: their headline success is coverage-only, not whole-episode grasp-valid success. |
| B4 | Low isolated-zero frequency rules against frequent isolated spikes; it does not validate all contiguous zeros or the polygon against physical dressing. Retain geometry checks and temporal scoring. |
| C1 | Reported diagnostic is about identifying the exact winner among candidates, not advantage over the policy. No causal learnability threshold established. |
| C2 | Collapse observed, random-walk explanation unproven. Evaluation seed blocks also change between checkpoints, so checkpoint variation mixes policy change with evaluation variation. |
| C3 | Four-step result supported locally; "nothing single-step resolves" is false even in its own 24-state sample, and no general affordability lower bound was measured. |
| C4 | Fixed-command divergence supported. Position equality is not proof of every restored state variable, and the contact-amplification attribution is too broad. |
| C5 | Strong local lead; not yet an action-ranking, matched-state or learning-speed comparison. |
| C6 | Present force signal is unvalidated; permanent exclusion of force from all inputs/rewards is unsupported. |
| D1 | Distribution mismatch and improved macro reconstruction supported. Gaussian norm percentile 1.000 is a rounded tail statistic, not zero support; Euler reconstruction error also reflects numerics. Local fitting gains did not establish better dressing. |
| D2 | The September 17 inventory reports eight compatible historical replays totaling 945,864 rows with `priv_dim=0`. Not true of all files now: clean state-run replay `replay_0010800/replay.json` has `obs_dim=35`, `priv_dim=35`, size 200,000. |
| D3 | Small-sample BC generalization failure supported. Imitation has no universal teacher ceiling; the FQL exception is also too small to establish robust improvement. |
| E1 | 0.046 is an early-state, in-sample oracle gap for seven macros, with acknowledged selection bias. It is not a validated learned-policy gain or an upper bound on other actions. Chance 1/7 requires a uniform null; use empirical class frequencies and tie-aware regret. |
| E2 | Local weak sensitivity under tested interventions is plausible; not a statement about all actions, latent directions, horizons or tolerance settings. |
| E3 | Incorrect random-action attribution: trained deterministic actor, `init_steps=0`; 0.52 is peak-event rate, final rate 0.08. |
| F1 | CLI omission repaired. A 35D geometric state helps isolate perception but is not perfect Markov information; changed timing prevents a clean ablation. |
| F2 | Two of five supplied bodies are held out by default in the relevant live-cell path; the split is logged. Make it explicit. Use zero holdout for the feasibility control, not to inflate a generalization score. |
| F3 | Run-dependent: current `train_sac` and `pretrain_wang` curriculum default is **0**, not 100. Record admitted transitions separately from simulated physics frames regardless. |
| F4 | Verified: 24,079.7 / 94,161.5 = **25.57%** evaluation time for the longest run. |
| F5 | Reported shared-directory contamination invalidates affected provenance. Preserve/quarantine; monotone CSV steps do not identify which process produced a checkpoint. |
| G1 | 15 peak-covered cells are neither an upper bound nor 15 valid successes. Missing category in the original decomposition: five final-covered cells with invalid grasp. |
| G2 | Systematic teacher failure on `tshirt_392` is supported. Tracking correlation does not establish whether grip design, geometry, material or control causes it. |
| G3 | Verified in longest-run evaluations: `hospital_gown` and `tshirt_68` have zero coverage success throughout. |
| H | Useful historical ledger, not a common-protocol leaderboard. The script's 733 seconds is simulation/evaluation runtime, not development time. |

Recomputed teacher categories are 6 final-covered/grasp-valid, 5 final-covered/
grasp-invalid, 4 covered-then-lost, and 10 never-covered: **25 total**. There are
15 peak-covered cells, of which nine retain the grasp over the recorded episode.
The stop study reruns four cells; 9/25 is the historical six plus three rescues,
not a fresh full-grid evaluation of the modified controller. Re-run the full grid
under the common contract before using it as a baseline score.

## 5. What the literature changes about the algorithm proposal

**RAL's stated core is established prior art.** Rollout Sampling Approximate
Policy Iteration (2008), sections 3–4, already simulates actions, accepts
statistically distinguishable labels, trains a policy classifier and allocates
rollout effort with bandit methods. It also discusses discarding uninformative
states. The 2010 rollout-allocation paper explicitly addresses fixed rollout
budgets. The claims that resettable simulators are almost unused in RL and that
confidence-gated rollout supervision has no counterpart should be withdrawn.
[RSPI](https://arxiv.org/html/0805.2027),
[rollout allocation](https://mohammadghavamzadeh.github.io/PUBLICATIONS/icml10-rollout.pdf).

**Potential shaping does not justify replacing the objective with milestone
reachability.** The policy-invariance result concerns adding
`gamma * Phi(s_next) - Phi(s)` to an existing reward under the theorem's
conditions. Greedily maximizing a short-horizon threading probability is a
different objective; it can prefer a grasp-invalid or unrecoverable state.
Finite episodes need terminal/time handling that preserves the telescoping
argument. A learned potential also must not silently change the task definition.
[Ng, Harada and Russell](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).

**Action chunks are a justified baseline, not new on their own.** Q-chunking
trains the critic on the executed action sequence and performs a corresponding
multi-step backup; the paper includes QC-FQL. This is more directly relevant
to the measured temporal effect than adding a novel name to single-action SAC.
Simply giving a single-action critic arbitrary off-policy multi-step returns
is not the same update.
[Q-chunking](https://arxiv.org/html/2507.07969v1).

**Learner-state supervision addresses a known distribution-shift problem.**
Labelling states visited by the current policy is useful but already central to
DAgger. Improved flow reconstruction is a representation result, not a policy
improvement certificate. The audited flow here decodes **one 6D action**, not
the 5x6 chunk assumed in the original FRS proposal.
[DAgger](https://proceedings.mlr.press/v15/ross11a.html).

## 6. Concrete next experiments, in order

### Gate 0: one versioned task and physical-time contract

Implement a shared episode scorer used by environment evaluation, SAC, FQL,
teacher and branch reporting. Preserve legacy fields under their original
definitions. Specify coverage geometry, whole-episode grasp history, simulator
errors, and completion held for a fixed **physical duration**, not an unqualified
12 decisions. For the historical 10 Hz reference, 12 decisions represents 1.2 s;
60 Hz would require 72. Keep peak progress as a diagnostic, not success.

Introduce an explicit hold/verified-completion phase. Do not end immediately at
the first threshold crossing and call that sustained success. A privileged
coverage-triggered stop is an oracle controller until a deployment-observable
stop detector is trained and evaluated. Do not silently require or remove the
separate `early_turn` paper filter.

Freeze a manifest of actual environment settings, body/garment splits, controller
rate, translation **and angular** speed, physical episode length, reward scale,
discount time and observation mode. Use `gamma(dt)=exp(-dt/tau)` for intended
physical-time comparisons, with consistent running rewards and entropy
conventions. Re-run teacher/teacher+hold and selected old checkpoints against
fixed evaluation seed blocks; retain a separate untouched final test split.

Protect each output directory with an exclusive process lock, record process
group/config/checkpoint identities and write checkpoints atomically. Keep
contaminated runs excluded. Record simulated physics frames, admitted replay
rows, updates and all elapsed costs separately. Reduce routine full-grid
evaluation frequency; reserve repeated full-grid tests for milestone/final
checkpoints. A high GPU utilization number is not the objective.

### Gate 1: spend a small fixed budget on numerical accuracy

Use existing recoverable states spanning approach, threading and elbow contact,
including both an easier cell and `tshirt_392`. Verify full-state restoration
as far as the API exposes it. Compare a 2x2 grid of Newton `{0.1, 0.01}` and CG
`{1e-2, 1e-4}` tolerances from common serialized physical states, not four
independently regenerated approach trajectories. Validate more accurate
reference solves on a subset; include initialization/warm-start effects.

Evaluate the incumbent plus a small frozen candidate set, with identical action
tapes for numerical diagnostics and an explicitly fixed continuation policy
for improvement tests. Measure mean/bias, repeat variance, **candidate-minus-
incumbent sign agreement with the reference**, grasp validity, solver failures
and isolated end-to-end wall time. Neither bitwise determinism nor the lowest
variance is the goal. Select the cheapest setting giving sufficiently reliable
useful comparisons. Start with a one-hour cap; an inconclusive cap result is
not evidence of impossibility. This does not require a solver rewrite.

### Gate 2: align the learning objective and data with completion

On the fixed contract, compare the original dense progress reward with a
completion objective that explicitly records valid grasp and verified hold,
optionally with correctly implemented potential shaping. This is an intentional
task-objective change and needs its own version and ablation. If whole-episode
grasp violation is irreversible under the chosen score, include that episode
state in the training contract rather than presenting identical observations
with incompatible success prospects.

Use successful existing episodes for initialization and failure episodes for
actual transition learning. Gather only missing on-policy recovery/completion
experience; do not recollect the entire corpus. Save geometry/state/history and
physical timestamps when collecting new data. Old rows without geometry cannot
be exactly relabelled for a new geometry reward. A teacher's counterfactual
action label is a BC label; it cannot replace the action in a stored RL
transition while retaining the old next state and reward.

First isolate control on one or two predeclared development cells with the
geometric actor; explicitly label this a feasibility test. Include a matched
point-observation actor only after this control works. Revisit history when
measured state aliasing warrants it. `tshirt_392` remains a separate failure
analysis stratum, not something to hide by averaging or tuning all test cells.

### Gate 3: one bounded RL comparison

With the preceding settings frozen, compare the same pretrained initializer
under single-action SAC/FQL, a held-action SAC control, and a proper chunk
actor/critic such as QC-FQL. Start with short physical durations suggested by
the 0.1 versus 0.4 s diagnostic, not an unqualified `H=4` copied between rates.
Keep gripper limits fixed; longer commitment trades action signal against lost
feedback and must be evaluated for grasp failures.

For an executed H-step chunk, the soft target has the structure

`sum_{j=0}^{H-1} gamma^j r[t+j] + gamma^H * (Q_target(s[t+H], A') - alpha log pi(A'|s[t+H]))`,

with the appropriate actual length and bootstrap mask for termination versus
time truncation. Define the entropy convention at chunk boundaries explicitly.
The critic must condition on the executed chunk. No claim is made that the
current single-action replay/flow already implements this.

Fix total workstation time, including offline preparation, queries, training
and evaluation. Use matched development cases and at least three training seeds
for a comparative claim. Start with a bounded pilot; require independently
evaluated full-episode improvement over the same initialized policy and a
competitive time-to-success curve before widening to the 25-cell grid. Local
score separation, lower BC loss and larger GPU batches are not pass criteria.

## 7. Where a contribution could remain

The useful research question is whether **numerical accuracy and action duration
can be allocated to obtain more reliable policy improvement per workstation
hour in contact-rich manipulation**. A candidate algorithm would compare against
the incumbent, retain multiple practically equivalent good actions, account
separately for sampling uncertainty and solver bias, and avoid spending all its
budget identifying an irrelevant unique winner. This is an unvalidated question,
not a novelty claim; multi-fidelity allocation and temporal abstraction require
a dedicated prior-art comparison before a paper contribution is stated.

Use fixed-precision chunked RL and conventional rollout policy iteration as
baselines. If those repairs solve the task efficiently, that is engineering
progress, not evidence of a new RL algorithm. If a distinct adaptive rule beats
them at matched total time on dressing and other contact tasks, then investigate
the contribution. IPC's plausible role is trustworthy offline/branch supervision;
it does not intrinsically improve SAC, require differentiating contact, or
establish real-world transfer without deployment evaluation.

## Reproduction and evidence boundaries

Run from the repository root:

```bash
PYTHONPATH=python /home/ge47gax/kun/genesis-world/.venv/bin/python \
  scripts/audit_dressing_training_inventory.py > /tmp/dressing_inventory_audit.json
```

The saved review output is
`output/uipc_manip/training_inventory_review_20260920/audit.json`. It contains
input SHA-256 hashes, timing/configuration comparisons, teacher categories,
signed calibration comparisons and the actual-function reward sweep. The
script is CPU-only; NumPy and SciPy are required. These artifacts are local
outputs, not additional committed training data.

Other directly inspected evidence: `train_sac.py::evaluate/resolve_defaults`,
`dressing_env.py::step/DressingConfig.max_translation`, `dressing_privileged.py`,
`scripts/ral_calibration.py`, `scripts/measure_predictability_horizon.py`,
`scripts/consequence_to_noise.py`, the stopping study's two `result.json` files,
the tolerance probe's two results, the historical data-inventory JSON and the
clean state-run replay metadata. The FRS, force and generalization results are
bounded readings of their existing reports, not new replications. No claim of
having independently rerun every historical experiment is made.
