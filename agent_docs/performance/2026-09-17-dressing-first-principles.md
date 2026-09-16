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

`scripts/evaluate_dressing_policies.py` restores the reference environment,
uses common observation seeds, reverses policy order in alternating rounds, and
records full episodes plus decision traces. `train_sac.evaluate` now preserves
the environment's grasp-valid success separately from geometry-only success.
The offline module and locality/evaluation changes passed 68 focused CPU tests
(one existing empty-distance warning in a simulator-error stub). Native policy
evaluation is the behavior check for the added rejection counters.

## Experiment results

The bounded offline runs and full dressing evaluations are recorded below after
completion. No performance claim follows from the loss curves alone.
