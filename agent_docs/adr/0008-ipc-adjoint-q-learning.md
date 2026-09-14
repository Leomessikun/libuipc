# ADR 0008 — IPC action-gradient supervision for off-policy critics

- Status: Accepted for the benchmark prototype; contact-conditioned extension remains proposed
- Date: 2026-09-14
- Owners: deformable manipulation research
- Implements: opt-in full-state SAC derivative loss and native direct-picker benchmark; see the implementation record
- Branch: `research/ipc-adjoint-q-learning`
- Supersedes: N/A; the existing physics actor experiment remains a separate baseline

## Decision

Develop **IPC Adjoint Q-Learning (IAQL)** in two stages. First establish a
correct, measurable implementation of action-gradient Bellman supervision on
a small deformable task. Then investigate **contact-conditioned IAQL**:
supervise only the local action directions for which the simulator derivative
has evidence of reliability. The actor keeps the SAC objective.

The research question is whether *reliable local mechanical derivatives*
improve off-policy policy learning enough to pay for their computation.
Neither the implicit-function identity nor adding a derivative loss to SAC is
a new contribution by itself. Successful dressing is a later application;
it is not the first correctness test.

Start with full-state observations and three translation actions. Progress to
visible point clouds and history only after the derivative and learning
experiments pass. Do not simultaneously add a privileged value network,
prioritized replay, actor regression to improved actions, and a new reward.

## Context: what the repository actually provides

| Existing component | Useful boundary | Gap for IAQL |
|---|---|---|
| [`SACAgent._update_critic`](../../python/uipc_manip/sac.py) | Twin scalar Q heads, soft backup, target networks | Currently value MSE only; backup is also clamped |
| [`FlatReplayBuffer`](../../python/uipc_manip/replay.py) | Replay, privileged fields, optional physics direction | Stored direction is not a versioned Bellman derivative; sequence batches have no equivalent physics fields |
| [`LinearSystemAdjointFeature`](../../include/uipc/diff_sim/linear_system_adjoint_feature.h) | Export and solve with the retained Newton matrix | Matrix is projected and from the last assembly, not necessarily the derivative at the accepted final state |
| [`physics_gradient_adjoint`](../../python/uipc_manip/physics_gradient_adjoint.py) | Multi-frame inertia chain, host export, translation tangent pass | Omits non-inertial cross-frame friction derivatives; tied to dressing layout |
| [`PhysicsActorSignal`](../../python/uipc_manip/physics_actor_signal.py) | Captured observation mapping and batched solver RHS | Last frame only; online deterministic continuation value; no complete reward/soft-Bellman derivative |
| [`tasks.py`](../../python/uipc_manip/tasks.py), [`GenesisIPCManipEnv`](../../python/uipc_manip/genesis_env.py) | `cloth_drag`, `cloth_fold`, `cable_drag` | Direct differentiable full-state observation adapter and replayable derivative capture are still needed |
| [`dressing_privileged`](../../python/uipc_manip/dressing_privileged.py) | Compact training features | NumPy features, discrete tests, no velocities; not a differentiable complete Markov state |

Current `cloth_drag` has 3 actions, five simulation frames per decision,
6 mm translation per action unit, and 10 ms frame duration. Its grasp targets
follow the commanded TCP; the robot also follows an IK/joint-control path.
Differentiate every path that actually influences cloth dynamics, including
robot contact if enabled. A direct-picker diagnostic variant may remove that
coupling explicitly, with the same variant used for every algorithm.

The existing [physics-gradient measurements](../performance/2026-09-13-physics-gradients.md)
already show that gradient quality depends on objective and contact regime.
Good opening-axis agreement does not validate a learned value gradient, the
full cloth response, or a six-dimensional Bellman gradient. A small residual
for the projected matrix does not establish derivative correctness.

## Prior work and claim boundary

Sources checked on 2026-09-14; this is a targeted comparison, not a claim of
exhaustive novelty clearance.

- **MAGE (2020)** explicitly trains critics for useful action-value gradients.
  This is an earlier direct precedent than JAVE.
  [D'Oro and Jaskowski](https://arxiv.org/abs/2004.14309).
- **First-order Sobolev RL (2025)** matches Bellman values and state/action
  derivatives, explicitly includes SAC, and discusses caching simulator
  derivatives in replay. Treat basic IAQL as its IPC instantiation and a
  required baseline, rather than claiming the loss or cache as new.
  [Schramm et al.](https://arxiv.org/html/2511.19165v1).
- **JAVE / Open-DiffLoco (2026)** supervises value Jacobians in critic-observation
  space using a learned observation-residual model. It already accommodates
  asymmetric actor/critic observations. Its appendix discusses bias from
  freezing the next-action dependence; “privileged training” alone does not
  distinguish our proposal.
  [Opat, Sections II-D and Appendix A](https://arxiv.org/html/2608.02069v1).
- **Distributional value gradients (2026)** studies joint value/gradient
  distributions and an off-policy actor-critic with a learned world model.
  This further limits broad claims about gradient-aware off-policy learning.
  [Paper](https://arxiv.org/html/2601.20071v1).
- **DiffIPC (2024)** establishes analytic adjoints for deformable contact and
  friction in PolyFEM. Its reported small overhead is specific to that
  implementation; it is not a runtime prediction for this backend.
  [Project and paper](https://huangzizhou.github.io/research/diffipc.html).
- **AHAC (2024)** adapts the differentiable horizon to reduce bias associated
  with stiff dynamics. Our proposed comparison concerns reliability weighting
  within sustained contact; no claim that an adjoint eliminates stiffness.
  [Project](https://adaptive-horizon-actor-critic.github.io/).
- **PODS (2021)** uses simulator-derived action gradients for policy improvement;
  **QAM (2026)** applies adjoint matching to Q-guided generative policies.
  Neither makes physics-derived critic supervision new by exclusion.
  [PODS](https://proceedings.mlr.press/v139/mora21a.html),
  [QAM](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0f85efb1e7545dc35a1b5e4d45aaf3c2-Abstract-Conference.html).

The candidate contribution is a validated method for selecting and using
reliable *action-space mechanical derivatives* under persistent deformable
contact, including their replay cost and observation limitations. This still
requires evidence beyond an unweighted Sobolev baseline and a simple residual
gate. If those baselines perform equally well, report the narrower result.

## One scalar function defines both targets

Let `s` contain the simulator's complete dynamic state: positions, velocities
or integrator history, moving boundaries, actuator state, material parameters,
and any friction/solver state that changes the transition. Fixed parameters
may be conditioned on rather than differentiated. Let `h` be the policy's
available observation history. Initially `h = s`.

Normalized replay action `u` lies in `[-1, 1]^d`. The actual controller maps it
to substep commands, including scaling, interpolation, rotations, and clamps.
The previous history and current state are fixed when differentiating in `u`.

For a single learning update, freeze the *parameters* of the target Q heads,
the actor used for the backup, and temperature. Match the existing SAC backup
by using the current actor, held fixed throughout construction of both targets.
Sample reparameterization noise `epsilon` once:

\[
u'=\tanh(\mu_\theta(h')+\sigma_\theta(h')\epsilon),\qquad
W(h';\epsilon)=\min_{j=1,2}Q_{\bar\phi_j}(h',u')
 -\alpha\log\pi_\theta(u'\mid h').
\]

For the full-state experiment, `h' = s'`. The raw target is

\[
Y(s,h,u;\epsilon)=R(s,u,s')+\gamma m W(h';\epsilon),
\qquad m=1-\mathrm{terminated}.
\]

Use the exact same reward scaling, discount, mask, target-head selection,
actor sample, and target transform for the scalar and derivative labels:

\[
y=\operatorname{sg}[T(Y)],\qquad
g=\operatorname{sg}[T'(Y)\,dY/du].
\]

Here `T` is identity or the existing scalar target clamp. For a clamped target,
the derivative is zero outside the interval; reject the derivative at its kink.
Log the clamped fraction. Do not pair a clamped scalar with an unclamped slope.
Any experiment removing the clamp must remove it for both SAC and IAQL.

Freeze network parameters, **not input derivatives**. Differentiate through
the next policy's mean, standard deviation, tanh, and log probability including
the tanh density correction. Detaching `u'` discards policy feedback. A target
`no_grad()` block cannot be reused unchanged for this construction. Only after
`y` and `g` exist are they detached from critic optimization.

For time-limit truncations use `m=1` and the saved pre-reset successor. True
terminal transitions use `m=0`. Treat masks and discrete mode choices as locally
fixed; a termination/success boundary is not covered by a smooth derivative.
With stochastic physics or sensing, differentiate a fixed-noise sample path
only where the pathwise derivative is valid. The sample gradient need not be
an unbiased derivative of the expectation at discontinuities.

This is the derivative of a **bootstrapped target**, not an oracle for
`Q^pi` or `Q*`. Exact mechanics still propagate errors in the continuation value.

## Implicit derivative over a complete control decision

Pack all unknown substep states into `X=(x_1,...,x_K)` and their actual residuals
into `F(X;s,u)=0`. Include lagged variables as state or as additional equations
when necessary. For a differentiable, converged root with nonsingular
`A = F_X`, let `ell(X,u)` be the raw target above, including observations,
velocity reconstruction, control bookkeeping, and all reward terms. Then

\[
A^T\lambda=\ell_X,\qquad
\boxed{g_{\rm raw}=\ell_u-F_u^T\lambda.}
\]

For one energy-minimizing substep, `A` reduces to its true energy Hessian.
Across the whole decision it is a block causal residual Jacobian, generally
not symmetric. Solve its adjoint by reversing the `K` substeps; never send
the entire nonsymmetric system to an SPD solver as if it were one Hessian.

For residuals `F_k(x_k,x_{k-1},x_{k-2},u)=0`, write
`H_k = F_{k,x_k}`, `B_k = F_{k,x_{k-1}}`, `C_k = F_{k,x_{k-2}}`:

\[
H_k^T\lambda_k=\ell_{x_k}
 -B_{k+1}^T\lambda_{k+1}-C_{k+2}^T\lambda_{k+2},\qquad
g_{\rm raw}=\ell_u-\sum_{k=1}^K F_{k,u}^T\lambda_k.
\]

Out-of-range terms vanish. The simple inertia model gives `B=-2M`, `C=M`;
the existing reverse pass implements that special case. Friction's dependence
on previous positions, lagged normals/tangents, and any moving geometry must
be added for a complete derivative. A terminal value depending on velocity
`v_K=(x_K-x_{K-1})/dt` contributes to *both* final positions. Per-substep reward
terms also seed intermediate right-hand sides.

Translation commands require `du -> metres` scaling and every interpolation
coefficient. Six-dimensional dressing adds the exact derivative of its rotation
parameterization at the executed rotation; a cross product at zero rotation
is not automatically the derivative of a finite Euler/axis-angle command.
Workspace clipping has a piecewise Jacobian. A command/execution norm ratio
does not recover a clamp's directional Jacobian. Reject unknown command paths.

The differentiation horizon is **one environment decision**, hence `K` implicit
solves per fresh scalar objective, not necessarily one solve. No policy rollout
or backpropagation across multiple RL decisions is required.

### Exact and approximate modes

An exact mode requires residual derivatives at the accepted root, all relevant
cross-substep/control terms, and a valid transpose solve. If the true Jacobian
is indefinite or nonsymmetric, the linear solver must support it; projecting
the forward Newton matrix to SPD does not preserve the implicit derivative.

The current feature instead provides `H_tilde`, a retained projected matrix.
Call the resulting label an **approximate IPC sensitivity**. Iterative
refinement improves the solve for `H_tilde`; it does not correct projection,
missing terms, wrong evaluation point, or discrete contact changes. A stricter
forward tolerance may reduce some errors but must be measured and costed.

For the full decision, if the true `A` is available and
`r_adj = ell_X - A^T lambda_hat`, the local error with otherwise exact inputs is

\[
g_* - \hat g=-F_u^TA^{-T}r_{\rm adj},\qquad
\|g_*-\hat g\|\leq\|A^{-1}F_u\|\,\|r_{\rm adj}\|.
\]

Using the projected residual alone omits
`(A_tilde-A)^T lambda_hat`. Errors in `ell_X`, `ell_u`, and `F_u` add further
terms. This motivates checking *action sensitivity*, not merely global
condition number or Newton iteration count. It is a local bound with explicit
assumptions, not a contact-wide correctness certificate.

## Critic and actor objectives

Use the existing twin scalar critic and SAC actor, initially in float32. Each
Q head's action gradient is obtained with `create_graph=True`, since optimizing
its derivative loss requires mixed derivatives with respect to action and
critic parameters. No simulator second derivative is required. Verify that
all participating encoder operations support this backward pass; do not detach
the action inside a dense point-cloud encoder to make the code run.

For `q_j=Q_phi_j(h,u)` and `g_j=grad_u q_j`, define

\[
L_Q=\mathbb E_D\sum_{j=1}^2(q_j-y)^2
 +\beta\,\mathbb E_{D_{\rm deriv}}\sum_{j=1}^2
 \frac{\|W_i^{1/2}(g_j-g_i)\|^2}{d\,\sigma_g^2}.
\]

`W_i` is a detached positive semidefinite action-space reliability matrix.
The first implementation uses `W_i=c_i I`, with binary validity for the basic
baseline. `sigma_g` is a positive scalar RMS scale fitted on the calibration
split and frozen; `u` already supplies the translation/rotation unit conversion.
Average over the sampled derivative batch, including zero-weight rows, so
rare acceptance does not silently strengthen each surviving label. An entirely
invalid derivative batch gives zero extra loss without affecting TD learning.

Squared derivative error is the primary baseline: it has a clear regression
target, permits zero slopes, and averages hidden-state labels in vector space.
Huber error is a robustness ablation. Direction plus log-magnitude matching is
deferred: cosine is ill-defined at zero and normalizing each label can distort
the mean under partial observation. There is no guarantee a bootstrapped
gradient operator is contractive just because `gamma < 1`.

Keep the existing actor and temperature objectives:

\[
L_\pi=\mathbb E_{h,u\sim\pi_\theta}
 [\alpha\log\pi_\theta(u\mid h)-\min_jQ_{\phi_j}(h,u)].
\]

Set the existing `physics_actor_weight=0` in IAQL runs. Supervision is at the
replayed action; do not apply a stored derivative as if it were measured at
a newly sampled actor action. Track actor-to-replay action distance and test
fresh policy actions separately. Start with uniform replay and identical
demo mix/update ratio to the control run.

## Contact-conditioned extension to test

Hard validity covers nonfinite output, solver failure, incomplete snapshot,
unknown command Jacobian, mismatched final observation, discrete boundary, and
excess adjoint residual. A finite residual alone is insufficient. Smooth zero
gradients remain valid labels. Contact presence alone never rejects a label.

Calibrate a reliability score from held-out *training* snapshots, spanning free
motion, stable contact, sliding, stick/slip transition, and jammed states.
Record normalized nonlinear/linear residuals, command regime, contact changes,
and repeat-simulation noise. Validate these predictors against finite differences
of the **same frozen Bellman objective**, not just contact energy. Use independent
states to measure both rejection rate and error among accepted samples.

The proposed stronger variant is directional trust. In a 3D or 6D action space,
some directions may remain useful while others cross a clamp or contact change.
Use a fixed orthonormal action basis (coordinate axes first). Estimate a
calibrated confidence `w_ik` for the slope in direction `e_k` and set

\[
W_i=\sum_{k=1}^d w_{ik}e_ke_k^T,\qquad 0\leq w_{ik}\leq1.
\]

A transparent first implementation fits a small binned table mapping solver
diagnostics and action-axis regime to the observed fraction of calibration
slopes meeting the error tolerance. Sparse/unseen bins get zero weight until
audited. Calibrate on several frozen critic snapshots so the score is not
validated only for one right-hand side. Per-transition finite differences are
an expensive reference variant; a predictor is useful only if it generalizes
to held-out states and evolving critics. Compare directional weighting to a
single scalar gate at matched accepted-label mass and compute budget.

Do not use agreement with the online critic as evidence of physical accuracy:
both can be wrong. Twin-target gradient disagreement measures continuation
uncertainty and may later modulate trust, but cannot certify mechanics either.
Minimum barrier distance alone cannot distinguish reliable sliding from an
unreliable transition. All trust weights are experimental, not guarantees.

## Replay and target freshness

An old `g = grad(R + gamma V_old)` does not become the derivative of a new
scalar target when the critic or policy changes. The old actor-signal replay
fields must not be silently interpreted as IAQL labels.

Keep normal replay for all TD updates. Maintain a bounded derivative sidecar
keyed by stable transition ID and ring-buffer generation. Sidecar eviction
never evicts the ordinary transition. The sidecar contains the exact pre-reset
successor, control and simulator configuration hashes, noise/snapshot identity,
derivative provenance, reliability diagnostics, and either of these backends:

1. **Reference adjoint refresh:** retain/reconstruct one decision's derivative
   tape and apply a fresh adjoint for the current target objective. Re-simulation
   needs integrator, friction, controller and RNG state, not positions alone.
   Check restored/recomputed endpoints and reward against the stored transition;
   if they disagree beyond the declared noise tolerance, skip its derivative.
   Do not pair one trajectory's scalar with another trajectory's derivative.
2. **Action-tangent cache:** for frequently reused transitions, solve for the
   `d` columns of the decision response once. Store `D = ds'/du` and the total
   immediate reward derivative; include all required successor state variables.
   For the full-state benchmark, refresh cheaply as
   `g = dR/du + gamma*m*D.T @ grad_s' W`, with the same clamp transform as `y`.
   Intermediate rewards require their accumulated derivative too. For history,
   include the direct action-to-history path. This is an optional compute/memory
   tradeoff, not a claimed new RL contribution.

The tangent cache is `O(state_dim * action_dim)`, not a full state Jacobian;
only the final label sent to the learner is `d` dimensional. Positions and
velocities for 3,529 cloth vertices with six actions require about 0.485 MiB
per transition in float32, about 497 MiB for 1,024 entries, before metadata
and other state. Never allocate it for the entire large replay by default.
Existing `tangent_pass` is a translation/inertia prototype, not this complete
state derivative. Keep physical derivatives separate from teacher-dependent
labels so Q/actor updates do not invalidate the mechanics cache.

Sample a uniform subset for derivative acquisition before trying priorities.
Match its state distribution to the TD buffer and preserve the ordinary demo
mixture; demo rows without replayable physics remain TD-only. If priority is
added later, retain nonzero uniform sampling and explicit importance weights.
Disagreement-based priority by itself is neither reliability nor novelty.

## Partial observation and privileged information

After the state-based test, let `h' = append(h,u,O(s'))`. Differentiate the
complete successor history map, including the explicit occurrence of `u`.
The current history is held fixed; no gradient is propagated to past actions.
Begin with fixed mesh samples and differentiable tool-relative coordinates.
Then use captured visible point IDs and voxel membership, following the existing
observation capture mechanism. That is a local fixed-selection derivative;
visibility changes and voxel changes remain an approximation to the rendered
observation. Tool coordinates and proprioceptive features have direct command
dependence and must not be omitted from the observation derivative.

A full simulator state supplies a *sample* slope for a visible history. It does
not uniquely determine the slope that a partially observed critic should learn.
If the belief `p(s|h)` is action independent, observations/history are sufficient,
and differentiation and expectation may be interchanged, the desired slope is

\[
\nabla_u Q(h,u)=\mathbb E_{s\sim p(s\mid h)}[\nabla_u Q(s,h,u)].
\]

Replay can violate this interpretation when a demonstrator used hidden state
to select actions: `p_D(s|h,u)` can differ from the deployment belief. A finite
history may also be insufficient. State-dependent reliability weighting changes
the conditional regression target further. Report these biases and compare
history lengths and actor observability at matched capacity; do not claim
that privileged labels remove partial observability. Average signed vectors,
including cancellation, rather than forcing every hidden-state direction to
agree with one observation.

A privileged `V_psi` is an optional later alternative. If used, **both** targets
must bootstrap from the same `V_barpsi(s',h')`; fit that value to the soft return
of the same observation-conditioned policy, with target lag and its own validation.
Scalar fitting of `V_psi` does not establish correct input derivatives.
The current compact NumPy privileged features cannot directly supply `dV/dx`.
Do not mix a visible-Q scalar backup with an unrelated privileged-V derivative.

At deployment only the actor and its visible-history construction remain.
The full-state benchmark alone provides no evidence of vision transfer.

## Update pseudocode

```text
collect with the current policy; save pre-reset transitions and optional physics sidecars
sample ordinary TD batch and uniform derivative subset with live sidecar identities
hold backup actor, target Q parameters, alpha and normalization fixed for this update
for each derivative row:
    verify snapshot/tangent provenance and complete action map
    sample next-policy noise epsilon once
    build Y = reward + gamma * terminal_mask * soft_target_value
    compute dY/du by complete-decision adjoint or cached action tangent
    apply the same scalar clamp and its derivative; obtain detached y and g
    obtain detached validity/reliability weights
compute ordinary scalar targets for the remaining TD rows
evaluate online Q1/Q2 and their action gradients with create_graph=True
optimize TD loss plus weighted gradient loss; teacher parameters receive no gradients
perform the existing delayed SAC actor/temperature updates and target updates
log derivative coverage, errors, target age, cache bytes and time by component
```

## Experiment sequence and stopping criteria

All thresholds below are proposed experiment gates, not observed results.
Calibration/test splits are by episode/start state, not correlated transitions.

### A. Numerical contract before cloth learning

Use a small linear implicit mechanical system with known matrices, at least
two substeps, a velocity-dependent terminal value, nonzero direct action cost,
and a differentiable stochastic next policy. Compare explicit tangent,
transposed adjoint and central differences of the same soft target. Sweep
finite-difference step sizes. Require relative error below `1e-5` in float64
away from kinks. Check terminals, truncation masks, action scaling, target
clamping, and a deliberately omitted substep/velocity path as a negative control.

The learner implementation must separately verify mixed action/parameter
gradients, detached teachers, both Q heads, zero/invalid labels, cache freshness,
and exact baseline equivalence with `beta=0`. Use existing test suites when
the corresponding production code changes.

### B. First IPC experiment: `cloth_drag`

Reuse the existing geometry, grasp and target sampler. Introduce a separately
named diagnostic configuration with a smooth reward and full dynamic state:

\[
d_\epsilon(c,g)=\sqrt{\|c-g\|^2+\epsilon_d^2},\qquad
R=w_p\frac{d_\epsilon(c_t,g)-d_\epsilon(c_{t+1},g)}{a_{\max}}
 -w_u\|u\|^2.
\]

Keep success as an evaluation metric and disable the existing discontinuous
success bonus for this configuration in *all* compared methods. Choose and
record one set of reward constants before comparisons. Do not silently use
smooth axis progress as the derivative of the original dressing reward.
Use a horizon of 150 decisions and initially retain the existing five substeps.

For physics checks start with friction disabled, then restore friction and
separately assess sliding and transition states. This isolates missing friction
terms. Select finite-difference steps in normalized action units, for example
`0.01, 0.03, 0.1`, and first measure repeated-action noise. Use a step only when
the paired objective difference exceeds five estimated standard errors and
neighboring step sizes agree. Log excluded/noisy samples, not just accepted ones.

On at least 100 held-out snapshots, proposed acceptance for a *magnitude-valid*
regime is median cosine at least 0.95 and median relative error at most 0.2,
with tenth-percentile cosine above 0.8. Evaluate all action axes and random
directions. Near-zero true slopes use absolute error; cosine is not meaningful.
Directions alone may pass while magnitude fails; those do not qualify for the
primary full-vector MSE experiment without a separate calibration/ablation.

First compare a frozen-dataset critic fit with and without derivative loss,
holding the backup networks fixed. Evaluate on held-out states and actions.
Only then run online SAC, so a failure can be attributed to mechanics,
regression, or bootstrapping rather than all three at once.

### C. Learning comparisons

| Variant | Purpose |
|---|---|
| SAC | Matched value-only control |
| SAC + finite-difference target gradients | Whether correct local slope labels help at all; expensive reference |
| IAQL with hard numerical validity only | IPC Sobolev baseline |
| IAQL + scalar reliability | Whether reliability predicts useful labels |
| IAQL + directional reliability | Candidate contribution beyond a scalar gate |
| IAQL + shuffled gradient labels | Check that improvement uses correct physical information |

Start with 3 paired seeds for debugging, then at least 5 independent paired
seeds for reporting. Tune `beta` on training/validation goals only; begin with
`{0, 0.01, 0.1, 1}` after fixed gradient normalization. Ramp from zero over the
first 10% of a declared training budget after replay prefill. Keep network,
reward, initialization, demos, UTD, temperature and evaluation protocol matched.
Count the calibration and finite-difference rollouts as training cost.

Report success and return against both transitions and elapsed training time,
including forward simulation, derivative capture/replay, solves, transfers,
neural mixed derivatives, and evaluation separately. Report peak memory,
accepted-label fraction by contact regime, TD error, held-out slope error,
head disagreement, action saturation and clamp fraction. A higher alignment
with the training teacher alone is not evidence of a better policy.

Evaluate counterfactual actions with a frozen continuation policy: compare
`u +/- eta * g/||g||` from identical snapshots, using paired noise and an `eta`
that stays inside the action box. Separately measure one-step frozen-target
differences (derivative correctness) and multi-step discounted returns
(policy usefulness). Use the matching soft objective when validating a soft
Q slope; report task success/raw return separately. A local Bellman derivative
does not guarantee the Monte Carlo return ordering. Report confidence intervals
and ties/noise rather than pre-assuming an 80–90% improvement rate.

### D. Escalate task difficulty only after B/C

1. `cloth_drag`: sustained plane contact and centroid control.
2. `cable_drag`: existing second morphology, different deformation response.
3. New cloth-strip-over-rounded-obstacle task: hidden distal marker, sustained
   sliding and geometric obstruction. This task is proposed, not implemented.
4. Fixed-geometry sleeve threading, then the existing dressing poses/garments.

The current `cloth_fold` marker is the held corner. Reaching its target does
not establish folding quality; do not use it as the main manipulation result
without an independent shape/self-contact metric.

Stop increasing task complexity if finite differences are unresolved, cached
labels disagree with fresh targets, correct derivative supervision does not
improve held-out slopes, or gains disappear at matched wall time. Diagnose the
failed stage before introducing another algorithmic module.

## Implementation sequence and consequences

1. Add a simulator-independent Bellman-target helper and twin-Q derivative loss
   to the existing SAC path, default off. Specify/reject unsupported categorical,
   sequence, precision, or encoder modes explicitly until tested.
2. Add complete-decision derivative capture, benchmark state/reward adapters,
   replay sidecar identity and target refresh. Generalize existing adjoint
   utilities; do not create another independent simulator/controller stack.
3. Run physics and frozen-critic gates, followed by the matched SAC experiment.
4. Add calibrated scalar/directional reliability only after the unweighted
   implementation is validated. Add visual history and dressing last.

Benefits sought: better action derivatives per expensive transition and local
contact credit without multi-decision simulator backpropagation. Costs:
additional solves/storage, critic double backward, derivative implementation
work, and continuing bootstrap/observation bias. No claim of exact contact
adjoints, global policy improvement, or a particular speedup is justified by
this design alone.

## Validation status

The initial change recorded a proposal and source/related-work audit. The
subsequent owner-authorized implementation and experiments are tracked in the
[benchmark record](../performance/2026-09-14-iaql-benchmark.md). Existing
physics-actor experiments remain separate from IAQL results.

Run the independent numerical check with NumPy installed:

```bash
python3 scripts/verify_iaql_design.py
```

The [check](../../scripts/verify_iaql_design.py) uses a 3-substep linear residual
system, terminal velocity, direct command dependence, intermediate state costs,
unequal action scales and a tanh Gaussian soft continuation. On Python 3 with
NumPy 1.26.4, all 24 combinations of mask, clamp, teacher version and difference
step passed: maximum relative finite-difference error `2.99e-10`, adjoint/tangent
agreement within `1e-12`. Deliberately retaining only the final frame produced
`6.78%` relative error. Reusing the old gradient after changing the continuation
gave an absolute gap `0.01317`; refreshing through the cached tangent matched
the new adjoint. The residual-error identity and bound also passed.
These are checks of the proposed equations, not of production IPC derivatives,
PyTorch double backward, stochastic discontinuities, or learning performance.
