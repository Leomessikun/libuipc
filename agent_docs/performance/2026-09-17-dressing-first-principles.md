# Dressing RL: objective, information, derivatives, and cost

Date: 2026-09-17. Scope: assistive dressing in the existing IPC environment.
The goal is a faster, more reliable learned policy, with a possible general RL
contribution. This record separates implemented controls from proposed research.
It does not declare a new algorithm successful or require another teacher corpus.

## What the evidence actually establishes

The [previous controlled continuation](2026-09-16-dressing-verified-results.md)
reused 125,016 transitions and Adam state. After 2,400 more transitions, repeated
two-cell development evaluations gave mean upper-arm coverage .47347 for SAC
and .37301 for the existing IPC actor term; both had zero successes in six
episodes. Training-loop times were 391 and 592 seconds. One training seed and
two repeated configurations do not establish a universal algorithm ranking.
The separate finite-correction test accepted zero of eight proposals for each
of IPC, SAC-gradient and random directions. Merely waiting longer on that
rejected correction scheme is not supported by these results.

The latest source audit found a concrete defect: `SACAgent._update` passed the
physics label into the dressing actor update **without the recorded action**.
The configured distance gate therefore never ran on this path. The full-state
benchmark had a different call path and did pass that anchor. The fix now passes
the replay action whenever physics labels are present. A regression test drives
the real point-replay update with valid but distant labels and verifies that
their physics contribution is zero. This fixes label locality, not label age,
critic accuracy, or the missing parts of the derivative. Historical experiments
remain measurements of their historical code, not results for the corrected gate.
Empty/gated actor batches now explicitly log zero physics rows/loss so retained
trainer metrics cannot display a prior batch's contribution as current.

## First-principles diagnosis

### 1. Accurate contact integration is not an exploration strategy

IPC changes the transition function. SAC still has to find commands that produce
useful cloth configurations. A small pull can increase tension without opening
a path around the elbow. A useful recovery might temporarily reduce progress.
This is a plausible explanation for the stalls, not a demonstrated classification
of every failed episode. Local optimization cannot guarantee discovery of a
different contact route. Literature on global optimization in differentiable
simulation documents related local-landscape limitations
([Antonova et al.](https://proceedings.mlr.press/v205/antonova23a.html)).

### 2. Our existing label is not the derivative of the deployed decision

Let `x` include physical and controller state, `a` be the policy command, `C`
be clipping/collision/tether logic, and `F` the six-substep IPC transition.
The actual next state is `x' = F(x, C(x,a))`. For the one-step objective

\[
 q(x,a)=r(x,a)+\gamma V(F(x,C(x,a))),
\]

its pathwise action derivative, within a fixed differentiable branch, is

\[
 \nabla_a q=\nabla_a r+\gamma
 [D_a(F\circ C)]^T\nabla_{x'}V.
\]

Here `r` is the decision reward, `gamma` the discount, and `V` the continuation
value. A soft SAC continuation would also have to use its own entropy objective
consistently. The production label uses the last assembled frame, cloth-position
derivatives of `min Q(o', mu(o'))`, fixed visibility/voxel choices and a fixed
tool reference. It omits the complete six-substep state chain, explicit reward
derivative, and controller/observation paths. Execution-fraction gating is not
differentiation of the collision and tether rules. It also does not check a
residual threshold, only whether the returned residual is finite.

The tool-reference omission is concrete: a visible cloth coordinate is
`p_cloth = x_cloth - x_tool`, so its action derivative is
`D_a x_cloth - D_a x_tool`. Static arm points have derivative `-D_a x_tool`.
The tool/goal observation tail changes too. Moving a held cuff and the tool
together can leave their relative coordinate nearly unchanged; a derivative
that holds the tool fixed misses this cancellation. Better IPC accuracy alone
cannot repair an incomplete observation/controller chain.

Thus a small adjoint residual certifies a linear solve, not this full policy
gradient. A finite-difference agreement test must perturb the **whole executed
decision**, including its controller and observation. Near a branch change,
local pathwise derivatives and finite changes need not agree. Even a correct
local gradient does not establish a useful finite-horizon policy improvement.

### 3. The IPC actor still depends on the critic

The solver maps a desired state change into an action direction. The desired
state change currently comes from a learned value gradient. Incorrect value
ranking can therefore defeat an accurate mechanical Jacobian. The historical
actor label is also stored while the critic and policy continue changing, then
normalized and scaled using a first-batch calibration. Reusing it indefinitely
does not make it a fresh gradient of the current objective. In the last matched
run only 2,400 of 127,416 replay rows carried valid labels before locality gating.

A post-training audit of that IPC checkpoint makes the mismatch quantitative:
only **31/2,400** labelled actions are within the configured .5 normalized-action
Euclidean radius of its current deterministic actor output. Median distance is
1.472; the 90th percentile is 2.079. The remaining 31 are **0.02433%** of all
127,416 rows. Under uniform sampling, a 64-row batch would contain at least one
such row with probability only about 1.55%. This is an endpoint audit, not a
reconstruction of each training-time gate. The six-dimensional radius itself
also needs calibration, particularly because x rotation is ignored by this
controller. It is not a demonstrated physical validity boundary.

Every labelled transition's accepted anchor translation matched the command
within float32 reconstruction error. Therefore this sample does not attribute
the old actor's failure to translation rejection. The actionable issues are
gradient/action mismatch, incomplete derivatives and learning from very sparse
usable labels. A future gradient actor should query the action it actually
improves, keep updates local, and record fresh-label cost separately from replay
TD learning. Fixing the gate is necessary but would mostly abstain on this replay.
Artifact: `dressing_first_principles_20260917/physics_replay_audit.json`.

Gradient mixing is already studied in
[AGPO](https://proceedings.mlr.press/v235/gao24m.html). A
[2026 re-examination](https://arxiv.org/abs/2604.18161) also reports that controlling
per-step variance can matter more than detecting bias in some robotics settings.
Consequently, neither “contacts make all gradients useless” nor “a confidence
gate makes our method novel” is justified. We need to measure the error and
variance of our actual estimator on dressing.

### 4. The observation can hide information needed for recovery

The current policy sees one partial point cloud and tool/goal features. It has
no explicit velocity history. The same visible shape can correspond to different
velocities or hidden cloth constraints. This is a plausible partial-observability
problem; it needs a controlled history ablation, not an assumption that adding a
transformer will solve it. The repo already contains sequence/history machinery.
Clipped x rotation is another nuisance: one of the six recorded action
coordinates is ignored by the current controller, but the actor still models it.

### 5. Dense progress and completed dressing are different objectives

The reward gives absolute arm progress each decision, with a stronger upper-arm
coefficient and geometric penalties. It is not a terminal-success reward or a
potential difference. A policy can improve return and still fail the .7 upper-arm
success threshold. This is not automatically a bug: it resembles the reference
task's shaping. Do not change the reward and attribute a gain solely to RL.
Report terminal coverage, success, valid-grasp success, failures and wall time.
Time limits currently retain bootstrapping; that corresponds to a continuing
task interpretation, which must be distinguished from a finite-deadline task.

The [original dressing work](https://arxiv.org/abs/2306.12372) uses pose-specialist
distillation and explicitly notes that its force threshold is in simulator
units, not calibrated newtons. Solver fidelity, material calibration, controller
behavior and real transfer are different claims. This research does not reopen
force channels as a default solution or assert demonstrated real-world transfer.

### 6. The expensive resource is useful new experience

Previous profiling attributed 96–98% of environment time to simulation. In the
small native timer window, collision-candidate detection took 56%, versus 11%
for PCG. More VRAM use or a higher reported GPU utilization is not proof of
faster learning. The target is success per wall-clock hour and per new IPC
transition. Existing trajectories can be optimized on the GPU without an IPC
step. That motivates an offline **initialization/control**, followed by online
improvement; it does not imply that offline RL can invent missing recoveries.

## Existing data and the implemented control

The selected source is `abl_dense_s1/checkpoints/checkpoint_00125016.pt` and
its paired `replay_latest`, preserving its dense/plain architecture. It contains
125,016 rows from five garments in region 13. These are policy experience, not
automatically successful demonstrations. It has no explicit episode identities.
Hashing observations and checking exact equality recovers 124,584 unique forward
links, all with row offset 24, and 432 contiguous segments of length 109–300
(median 300). Duplicate starting observations: zero. Ambiguous destinations,
multiple predecessors, backwards links and terminal predecessors are refused.
These are observation-contiguous segments, not certified physical snapshots.
No held-out expert rollouts are added to training.

`uipc_manip.offline_rl` implements three controls on the same saved data: IQL,
behavior cloning (BC), and replay-only SAC. IQL is an established algorithm
([Kostrikov et al.](https://arxiv.org/abs/2110.06169)), not our contribution.
It uses:

\[
 L_V=E[|\tau-1_{Q^- - V<0}|(Q^- - V)^2],\quad
 y=r+\gamma m V(o'),\quad L_Q=\sum_i E[(Q_i(o,a)-y)^2],
\]
\[
 L_\pi=-E[\min(e^{\beta(Q^- -V)},100)\log\pi(a|o)].
\]

`Q^-` is the minimum of target critics at the recorded action, `m` the existing
bootstrap mask, `tau=.7` the expectile and `beta=3` the inverse temperature.
The actor and dense point-cloud critic are reused. The independent observation
value copies the saved critic encoder and Q1 head with a constant zero-action
input. Policy learning uses recorded-action likelihood; it never differentiates
Q with respect to an action. BC replaces the weights by one and freezes the
critics. SAC is a replay-only diagnostic, not a recommended offline algorithm.

All arms start from the same saved weights and **fresh optimizers**. This is
explicitly an offline objective switch, not the preserved-Adam continuation
above. Each uses 2,000 minibatches of 128, with the same segment split and seed.
SAC retains its every-fourth-update actor schedule; IQL/BC update the actor every
batch. Therefore both optimizer counts and wall time must be reported. Validation
segments are excluded from these new updates, but the source SAC checkpoint
already saw the corpus: this is adaptation validation, not an untouched test.

The exported file can be evaluated with the existing policy interface. An IQL
critic estimates an unregularized return, whereas SAC uses a soft return. Do not
silently resume its critic/optimizer as a matched SAC continuation. The saved
metadata marks the offline objective, and the separate IQL value artifact retains
its weights and optimizer. An integrated online IQL trainer is not implemented.

## Design of the next algorithm, with novelty limits

The preferred research architecture has three parts:

1. A history-conditioned distribution over short action sequences, initialized
   from existing trajectories, so alternatives around the elbow can remain
   distinct instead of averaging their commands. Sequence replay must preserve
   actual executed commands and boundaries.
2. A critic conditioned on the **entire executed sequence**, with target
   `sum(gamma**j * reward[j]) + gamma**H * bootstrap_value` for a chunk of `H`
   decisions. Do not attach that return to only its first action and call it
   an unbiased off-policy backup. This principle is already
   [Q-chunking](https://arxiv.org/abs/2507.07969). Flow policies and efficient
   policy extraction are already developed in
   [FQL](https://arxiv.org/abs/2502.02538). Adaptive contact-dependent commitment
   also has prior work, including the
   [AQC preprint](https://arxiv.org/abs/2605.05544). Combining those names is not
   by itself a novel contribution.
3. An IPC-specific local improvement mechanism that differentiates the complete
   controller/transition contract, preserves action locality, and predicts when
   a finite correction leaves its valid contact branch. The hypothesis is that
   this can improve an already viable sequence more cheaply than extra sampled
   rollouts. It must beat the **same sequence policy without IPC supervision**
   at equal total compute, including labels and verification. A deployable actor
   must infer from available observations/history; privileged mechanics may be
   used for training, not silently required on the robot.

Only item 3 is a candidate contribution specific to this project, and its novelty
is not established. The previous first-action verifier already failed its small
test and incurred substantial rollout overhead. Do not simply repeat it at a
larger budget. First establish a viable data-derived policy, identify the actual
controller/contact branch in failures, and validate the full derivative against
finite perturbations. Fixing the missing replay anchor is necessary but cannot
substitute for that work.

The owner's clarification is binding: offline-to-online learning is not solved
by this proposal, and IPC integration has not been ruled out. The offline run is
a bounded diagnostic. Gradient-free IPC mechanisms also remain possible:
privileged value learning, execution prediction, and recovery supervision from
forward simulation. Before choosing among them, use the new decision traces to
separate collision-rule refusals, tether refusals, poor grasp tracking and
executed-but-unproductive motion. A contact penalty alone would conflate necessary
contact with failure. The decision trace labels measure controller acceptance,
not cloth motion or proof of a particular hidden contact topology.

### Actor-side follow-up

The owner specifically asked whether the actor route survives a weak critic.
Yes, but the existing physics actor is not critic-independent: it differentiates
the learned next-state value. An alternative is to improve a short sequence
against simulated task outcomes and then fit the actor to successful changes.
That can use derivatives on validated branches or forward search otherwise.
The objective must retain useful elbow passage and grasp validity; merely
reducing contact or maximizing a local geometric proxy is insufficient.

There is concrete precedent for direct differentiable cloth policy training:
[DiffCloth](https://www.csail.mit.edu/research/diffcloth-differentiable-cloth-simulation-dry-frictional-contact)
demonstrates a closed-loop hat controller as well as dressing trajectory
optimization. This is evidence of feasibility in a related simulated setting,
not evidence of success for our sleeve/controller/observation contract.
Its [released controller](https://github.com/omegaiota/DiffCloth/blob/master/src/python_code/hatController.py)
reads the full cloth position difference from a target and velocity/geometry
features; it is not a drop-in partial-point-cloud deployment policy.
Trajectory optimization followed by policy learning also has longstanding
precedent in [Guided Policy Search](https://proceedings.mlr.press/v28/levine13.html).
It must not be presented as new just because IPC supplies the simulator.
[AHAC](https://arxiv.org/abs/2405.17784) and SHAC are relevant to a direct
short-horizon gradient route, but still use a learned continuation value and
do not make our partial last-frame derivative complete.

The actor-side acceptance sequence is therefore: repeatable finite recovery
on development elbow states; measurable policy learning from that recovery;
full dressing improvement on separate configurations; then a compute-matched
comparison with SAC and the identical recovery method without IPC derivatives.
Passing only the first gate is trajectory optimization, not an RL result.
The prior 0/8 first-action correction result fails the first gate for that exact
proposal scheme. No new expensive sequence-search campaign is launched here.

`scripts/evaluate_dressing_policies.py` restores the reference environment,
uses common observation seeds, reverses policy order in alternating rounds, and
records full episodes plus decision traces. `train_sac.evaluate` now preserves
the environment's grasp-valid success separately from geometry-only success.
The offline module and locality/evaluation changes passed 68 focused CPU tests
(one existing empty-distance warning in a simulator-error stub). Native policy
evaluation is the behavior check for the added rejection counters.
An additional SAC/trainer regression run exposed a pre-existing flaky assertion
that four samples with replacement must contain both episode-step values. The
test now verifies both values in stored replay and checks the sampled padding
against each sampled step. The 59-test SAC/trainer run then passed. The unrelated
ADR archive test still fails because ADR 0008 lacks its required Consequences
heading; the performance-index/link check passes.

## Experiment results

All planned runs finished. No further training/search was launched after this
bounded comparison. Each offline arm uses 100,171 training rows and 24,845
adaptation-validation rows, batch 128, 2,000 updates, seed 1. Training uses no
new simulator transitions. Setup/loading took approximately 13 s per arm.

| Policy | Offline training seconds | Actor updates | Mean terminal coverage | Two round means | Successful / evaluated episodes |
| --- | ---: | ---: | ---: | --- | ---: |
| Starting SAC checkpoint | 0 | 0 | .16101 | .26654, .05547 | 0/4 |
| IQL adaptation | 220.24 | 2,000 | .13509 | .06729, .20288 | 0/4 |
| BC adaptation | 67.24 | 2,000 | .18434 | .18130, .18738 | 0/4 |
| Replay-only SAC | 137.20 | 500 | .22304 | .09937, .34671 | 0/4 |

Evaluation used tshirt_26/body14046 and body14049, 300 decisions each, two rounds,
common seed bases 1097/1194, and reversed policy order in the second round.
These are the previously used development cells, not a final untouched test.
All 16 episodes ended without simulator errors, and all had zero grasp-valid
success as well. Evaluation consumed **4,800 transitions and 637.12 s**, plus
5.31 s world construction; these costs are additional to the offline training
times above. Counts exclude setup/reset hold frames.

The apparent mean coverage gains of BC and replay-only SAC do not establish
better policies. Over 1,200 evaluation decisions per policy, invalid-grasp
decisions numbered **0 / 23 / 497 / 582** for source/IQL/BC/SAC-replay. Collision
refusals were **0 / 0 / 2,133 / 3,088** substeps, and tether refusals were
**0 / 0 / 1,636 / 331**. Those counters can overlap; they are not disjoint failure
probabilities. The added counters passed range/accepted-motion consistency checks
on every recorded evaluation transition.

Two distinct failure mechanisms are visible:

- The original policy on body14049 accepts all commands and retains a valid
  grasp, but in the last 60 decisions moves its anchor 6.0/14.0 cm in cumulative
  path length while gaining only .0062/.0023 coverage. This is executed but
  unproductive motion; the traces alone do not distinguish snag topology from
  an ineffective direction.
- BC on body14046 ends at zero coverage both rounds. Its last 60 decisions
  command 48.0/46.7 cm cumulative translation and achieve zero anchor movement.
  Here controller rejection is a real failure mode, and treating commands as
  executed actions would be wrong. More coverage on the other body does not
  repair this failure.

Source-body14046 ends at .406 coverage in one round and zero in the other,
despite valid grasps throughout. Its two first actions differ by only about
6.7e-6 in normalized action norm, then trajectories diverge. The provenance of
the divergence is not isolated here; neither observation randomness nor IPC
nondeterminism alone is established as its cause. Several coverage traces also
show abrupt geometric-metric transitions. These observations preclude confident
algorithm ranking from one or two rollouts and motivate inspecting the cloth
configuration at metric collapses before modifying the reward.

**Decision:** this warm-start IQL configuration provides no basis for a large
offline-to-online study or a novel-algorithm claim. It does not disprove offline
RL generally, or establish that these are the best data/architecture/training
settings. The current comparison changes several learning components and is not
a causal proof that the critic gradient is the sole problem. Retain the existing
online SAC continuation as a reference, and make the next IPC actor experiment
address action-locality and the complete executed/observed derivative contract.
If reliable local labels still do not produce recoveries, test task-based
sequence supervision without relying on the critic's action/state gradients.
Neither route should be called a strong new RL algorithm until full dressing
success and matched-cost improvement are measured.

Final focused regression: **88 passed**, with one existing empty-distance warning
in a simulated-error test. No simulator/force tolerance was changed. The separate
ADR archive heading failure noted above is unrelated to this implementation.

Artifacts under `output/uipc_manip/dressing_first_principles_20260917/`:
`iql/`, `bc/`, `sac/` contain objectives, timing, checkpoints and loss logs;
`data_audit.json`, `recovered_links.npz`, `physics_replay_audit.json` retain the
data/label audits; `evaluation.json` retains every decision and episode record;
`summary.json` and `coverage.png`/`coverage.pdf` summarize results. The plot's
thin lines are individual rounds, not confidence intervals. `summarize.py`
regenerates the aggregate artifacts.

Reproduction (with the repository's Genesis venv and normal native library paths):

```bash
python -m uipc_manip.offline_rl \
  --checkpoint output/uipc_manip/abl_dense_s1/checkpoints/checkpoint_00125016.pt \
  --replay output/uipc_manip/abl_dense_s1/checkpoints/replay_latest \
  --mode iql --steps 2000 --batch-size 128 --out NEW_RUN
# Repeat with --mode bc / sac in separate directories.
python scripts/evaluate_dressing_policies.py \
  --reference output/uipc_manip/abl_dense_s1/checkpoints/checkpoint_00125016.pt \
  --policy source=SOURCE_CHECKPOINT iql=IQL_CHECKPOINT bc=BC_CHECKPOINT sac_replay=SAC_CHECKPOINT \
  --cells tshirt_26:14046 tshirt_26:14049 --rounds 2 --out NEW_EVALUATION.json
```
