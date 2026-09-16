# Dressing RL redesign: verified local policy improvement

Date: 2026-09-16. Status: research proposal, not an implemented or validated algorithm.
Scope: learn one deployable assistive-dressing policy using the existing trajectory
corpus and expensive IPC simulation. Evaluate other tasks after the dressing
mechanism is established. The owner stopped the cloth-drag study; do not resume
it as a prerequisite for this work. No new training was launched for this report.

**Implementation follow-up:** the bounded frozen-critic diagnostic is now
implemented and tested on two dressing cells. It accepted no IPC corrections
(0/8 anchors). Replay/optimizer-preserving continuation and actual dressing
profiling were added separately. See [the experiment record](2026-09-16-dressing-verified-results.md).
The full online algorithm below remains a proposal; the original research-pass
status above describes when this document was written.

## 1. What the evidence actually establishes

The dressing and cloth-drag experiments use different environments, observations,
action spaces, derivative paths, and actor updates. Their success rates are not
interchangeable.

| Evidence | Finding | What it cannot establish |
| --- | --- | --- |
| Dressing `pg_phys_s1`, 24k additional transitions | Upper-arm coverage .14581 to .30905; final success 6/25 | Robust superiority across seeds |
| Dressing `pg_ctrl_s1`, same budget | Coverage .16039 to .04299; final success 0/25 | Performance of a stable SAC fine-tuning baseline |
| Dressing frozen-state probes | Dense critic + physics direction passes some elbows where that critic's action derivative retreats | Generality to all critics, states, or an evolving learner |
| Twelve-decision dressing trajectory optimization | Most gain over the expert came from the larger action scale; matched-scale optimization added little | A reliable long-horizon derivative or learned release/advance sequence |
| Cloth-drag fresh study, 30,016 transitions, paired seeds 1–4 | SAC 82.0% vs IPC 69.5% success | Dressing or elbow success |
| One-update cloth-drag probes | Actual displacement and optimizer state materially change outcomes | Stable continuous policy learning |

Dressing evaluations used 25 held-out garment/body cells, 300 decisions, and the
configured upper-arm success threshold .7. One rollout per cell is not a robust
success-probability estimate. The physics and control runs both started from
`abl_dense_s1` at 125k updates, with empty replay and fresh Adam; both suffered
watchdog trips. This baseline reset is a confound to resolve before claiming a gain.

Local sources:

- [Dressing physics experiments](2026-09-13-physics-gradients.md), especially
  Levels 3.1–3.4. The older concluding sentence saying no policy has been trained
  is superseded by the Level 3.4 training results.
- [Actor-interface measurements](2026-09-16-actor-interface-results.md).
- [Fresh benchmark protocol](2026-09-15-fresh-ipc-actor.md).
- [Force audit](2026-09-12-force-learning-audit.md) and
  [force prediction](2026-09-12-force-in-training.md).
- Raw evaluations: `output/uipc_manip/{pg_phys_s1,pg_ctrl_s1}/eval_log.csv`.

## 2. Newly checked implementation distinctions

The production dressing signal in `python/uipc_manip/physics_actor_signal.py`
uses the last frame's assembled system to pull back the gradient of
`min Q(o_next, mu(o_next))`. Visibility and voxel membership are frozen. The
autograd graph varies cloth positions while the tool reference and other
observation fields are held fixed. This is a restricted local surrogate, not the
complete derivative of a dressing decision, rendered observation, or soft return.
It omits an explicit immediate-reward derivative and entropy in this signal.
These facts do not invalidate its measured usefulness as a direction.

`sac.py::physics_actor_loss` normalizes stored directions and applies a linear
actor objective, with beta calibrated once. In contrast, the newer full-state
`iaql.py::soft_targets_detailed` combines reward and continuation derivatives,
entropy, twin-critic disagreement, and fresh-action handling. Those benchmark
improvements have not automatically become a validated visual dressing algorithm.

The dressing records identify additional boundaries:

1. The upper-arm progress signal is locally flat before elbow passage at the
   tested states. A one-step progress gradient alone cannot resolve that phase.
2. A weaker critic reversed the useful physics directions. Accurate state
   sensitivity does not repair an incorrect value landscape.
3. Visibility changes and critic roughness made finite action changes disagree
   with infinitesimal value derivatives. A small linear-solve residual does not
   measure the quality of the resulting policy update.
4. The historical multi-decision linearization largely reduced to a static
   response and missed contact-path dependence. Extending that chain is not an
   evidence-based default.
5. One stall was locked by the environment's command-rejection rule near the arm.
   Differentiating cloth physics does not differentiate that discrete command map.
   Preserve the rule in evaluation; do not silently relax it to improve scores.

These are measured or source-visible limitations. Which one dominates ongoing
dressing learning remains unknown.

## 3. Existing data and interfaces to reuse

Confirmed `abl_dense_s1/checkpoints/replay_latest/replay.json`: 125,016 stored
transitions, observation dimension 5,383, action dimension 6, reward scale .5.
Other replay snapshots exist for the teacher and both physics/control fine-tunes.
The corpus already exists; this proposal does not require another large expert
collection pipeline or the 27-teacher chain.

A replay tuple is not a restorable simulator state. For derivative queries and
counterfactual rollouts, require a compatible native world dump plus Python
environment bookkeeping. `physics_gradient_probe.take_snapshot/restore` restores
the anchor, held offsets, progress, episode clock, heuristic state, and native
frame, and measures position restoration error. Historical JSON measurements
alone do not prove that every old dump remains restorable in the current binary.
Recover selected diagnostic states with the existing reproducer when necessary;
do not claim that replay arrays alone provide a simulator reset.

Freeze whole garment/body/pose splits before any new selection or tuning. The
historical 25-cell set has already been inspected repeatedly and should be called
development evaluation for the redesign. Reserve a separate final test split.

## 4. Research hypothesis and candidate algorithm

**Hypothesis:** IPC derivatives are useful as occasional action proposals near
difficult interactions, but finite-rollout verification and bounded policy fitting
are necessary to convert those proposals into robust improvements at acceptable
cost. This hypothesis may be false; the controls below are designed to reject it.

Maintain the existing dense/residual SAC policy and critic. Begin from a validated
existing dressing checkpoint and available replay. Keep the policy's deployment
inputs unchanged. The simulator and derivatives are training-only resources.
Do not require an IPC solve at robot inference time.

### 4.1 Generate a local proposal at the current action

For a restorable state `s` and current policy action `a0`, evaluate a physics
direction at that same action. The initial implementation may reuse the historical
last-frame direction as an explicitly approximate proposal; compare a complete
one-decision chain only where its cloth-body derivatives have been validated.

Generate `a_plus = a0 + delta` inside the existing action bounds. Scale translation
and rotation separately in physical units; report both requested and executed
movement. Do not reuse the cloth-drag .03 radius as an unexamined dressing scale.
Set a radius on development states using repeatability and finite-step tests,
then freeze it for held-out comparisons.

Evaluate a movement-matched SAC direction, a random direction, and optionally the
negative physics direction on the same snapshots. Test the current policy action
as the reference. Both zero and nonfinite derivative cases must abstain rather
than create arbitrary normalized labels.

### 4.2 Verify finite improvement with the actual environment

Apply one candidate first action, then use the same frozen policy in closed loop
for the remainder of `H` decisions. Start with `H=12`, matching existing dressing
probes; this is a diagnostic choice, not a proven sufficient horizon.

For each candidate, measure the paired truncated return difference

`A_H(s, a) = sum_{t=0}^{H-1} gamma^t [r_t(candidate) - r_t(reference)]`.

Use the same reward scale, observation pipeline, action-rejection rules, reset
state, and policy-noise sequence in paired branches. Record terminal upper-arm
coverage, actual elbow passage, grasp validity, refusal rate, and simulator
failures separately. Repeated reference branches establish the numerical noise
floor. Select candidates on one repeat set and confirm on independent repeats
to reduce winner's bias. A small number of repeats is a numerical check, not a
population-level statistical guarantee.

Accept only an improvement above that noise floor without regression in the
predeclared task endpoints. Do not accept solely because the same critic that
generated the direction predicts a larger value. Keep terminal critic values as
diagnostics in this first gate. This gate evaluates a finite-horizon surrogate;
it does not guarantee infinite-horizon improvement or preservation of soft SAC's
objective. Full-episode tests remain necessary.

If all branches remain flat, abstain. This explicitly acknowledges that a short
one-action intervention may not discover release-then-advance. A later action-chunk
variant must give both SAC and IPC the same chunk action space and sequence prior,
then use the matching multi-step Bellman target. Do not validate an open-loop
sequence and label its first action as successful under an incompatible policy
continuation. No long-horizon backpropagation is required by the initial design.

### 4.3 Fit accepted actions while retaining SAC learning

Store accepted tuples `(observation/history, a0, a_plus, A_H, radius, policy_version)`
in a bounded recent correction buffer. Store executed transitions from every
branch, including failures, in ordinary replay with correct reset/terminal flags.
Avoid optimistic filtering of the TD replay itself.

Use the existing SAC objective plus a bounded supervised correction:

`L_actor = L_SAC + lambda * E[w * ||mu_theta(h) - a_plus||^2]`,

where `w = clip((A_H - noise_margin)/return_scale, 0, w_max)` is detached and its
scaling uses only training data. This extra loss is intentionally a biased local
policy-improvement surrogate. It is not an unbiased gradient of the original SAC
objective. First validate lambda and weight scaling on development data, then
freeze them; do not publish a single tuned seed as a general algorithm result.

Enforce a bound on the *actual* action displacement after the total update on
both correction states and a representative replay anchor batch. Restore model
and optimizer state when retrying an oversized update. This is a sampled guard,
not a global KL or performance guarantee. Use the same guard and update budget
for matched controls. Existing probes already show that target regression alone
does not solve inherited-momentum or network-coupling problems.

Expire corrections when the policy has moved beyond the verified neighborhood;
old verified returns do not remain valid under a new continuation policy. Keep
ordinary TD replay independent of this expiration. Start with a fixed, sparse
query budget rather than training an additional gating network.

### 4.4 Make speed a measurable requirement

Count simulator transitions used for candidates, reference branches, repeats,
state recovery, and ordinary collection. Record derivative, simulation, observation,
and learner time separately. Compare success versus both total simulator work and
exclusive elapsed/GPU time. All methods receive the same data access and either
the same total query budget or the same wall-time budget, with both views reported.

This candidate could be slower than SAC. Reject it as a speed contribution if
verification cost exceeds the saved training cost. Querying only a selected subset
of training states is a proposed cost control, not a measured speedup. Select
states with a mixture of recent low-progress states and uniform samples, so the
algorithm cannot improve its apparent reliability by avoiding difficult states.
Elbow labels are for stratified diagnosis, not a required input to a general RL rule.

## 5. Decisive dressing experiment before another long run

First implement a bounded extension of the existing dressing probe/fine-tune
tools, not a new cloth-drag experiment or a second training framework.

1. Reproduce same-state reference rollouts at approach, elbow, refusal/lock, and
   post-elbow states. Include development and held-out garment/body cells. Failed
   restoration or simulator failures invalidate the comparison and remain reported.
2. Freeze the existing dense actor/critic. Compare current policy, SAC proposal,
   IPC proposal, and matched random proposal with identical action constraints.
   Use an equal candidate budget. Report gains and harms by state category,
   numerical repeat spread, and complete cost.
3. Fit accepted proposals, then evaluate the *network's* rollouts from separate
   states. This isolates useful proposals that the actor fails to absorb.
4. Only after the local mechanism passes, run a bounded dressing fine-tune with
   at least three paired training seeds; five is preferable for a stronger claim.
   Same initialization, replay, observations, reward, actor/critic architecture,
   optimizers, and update budget. Establish a healthy continuation baseline before
   introducing the correction; do not repeat the unexplained empty-replay collapse.
5. Evaluate complete 300-decision dressing episodes, including initial approach.
   Report final success, mean and worst-group coverage, elbow-passage success,
   simulator/watchdog failures, executed-action fraction, and total training cost.
   An elbow-reset curriculum is not evidence of start-to-finish dressing unless
   the final policy is also evaluated from the original initial-state distribution.

Required ablations: matched SAC; historical raw physics update; verified SAC
proposals; verified random proposals; verified IPC proposals. If verification
alone explains the gain, do not attribute it to IPC derivatives. Compare to a
short-horizon analytic actor-critic baseline such as SAPO when its required
derivative paths are available; disclose any approximation instead of calling a
partial implementation a faithful reproduction.

Stop rules:

- IPC proposals lose to SAC/random at equal cost: reject the proposed derivative
  channel for those states; investigate reward/continuation/observation derivatives.
- Proposals help but the fitted actor loses: investigate optimization and partial
  observability before generating more labels.
- Short rollouts help but complete dressing loses: the horizon/objective is wrong;
  the result is not successful dressing learning.
- Verified IPC gives no advantage over verified random/SAC: there is no evidence
  for a physics-specific algorithmic contribution.
- More samples saved but more total time spent: report sample efficiency only.

## 6. Literature and novelty boundary

| Primary source | What to borrow | What it means for our claim |
| --- | --- | --- |
| [SVG (2015)](https://arxiv.org/abs/1510.09142) | One-step dynamics derivative plus learned continuation | The basic derivative decomposition is established |
| [SHAC (2022)](https://arxiv.org/abs/2204.07137) | Short differentiated horizon and terminal value | A relevant baseline, not permission to trust our old long chain |
| [SAPO (2025)](https://arxiv.org/html/2412.12089v2) | Maximum-entropy analytic actor-critic for expensive deformable simulation | Closest broad motivation; uses on-policy value learning; its particle-cloud inputs do not validate our visibility/voxel derivatives |
| [AHAC (2024)](https://proceedings.mlr.press/v235/georgiev24a.html) | Adapt differentiation near stiff dynamics | Contact-aware horizon selection alone is established |
| [Suh et al. (2022)](https://arxiv.org/abs/2202.00817) | Bias/variance pitfalls of differentiable policy gradients | Exact local derivatives need not be good finite-sample estimators |
| [Onoda et al. (2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4f0a2a0b2ca6ffd5c8d5de26d3e8d54d-Abstract-Conference.html) | Estimator switching and per-step variance control | Do not assume contact classification is the main missing ingredient; its estimator mixture is not a theorem for SAC gradients |
| [DiffMJX (2026)](https://arxiv.org/abs/2506.14186) | Separate physical forward dynamics from informative backward gradients | A possible later surrogate-gradient ablation; no established IPC portability |
| [Guided policy search (2014)](https://proceedings.mlr.press/v32/levine14.html) | Optimize behavior and constrain policy agreement | Physics-guided targets and policy fitting alone are not novel |
| [DiffCloth (2022)](https://people.csail.mit.edu/liyifei/publication/diffcloth/) | Differentiable frictional cloth control and dressing optimization | No first differentiable dressing claim |
| [DiffSkill (2022)](https://arxiv.org/abs/2203.17275) and [DexDeform (2023)](https://arxiv.org/html/2304.03223) | Existing demonstrations can initialize useful manipulation sequences; differentiable refinement has local-minimum limitations | Data-guided refinement and sequence priors are established, optional extensions rather than the initial implementation |
| [CPDeform (2022)](https://arxiv.org/abs/2205.02835) | Contact switching can defeat local gradient optimization | Elbow sequence failure cannot be assumed solvable by better gradient magnitude |
| [One Policy (2023)](https://arxiv.org/abs/2306.12372) | Deployable visual dressing policy, dense Q, diverse garments/poses | Keep the actual task and policy-input contract central |
| [FMVP (2025)](https://arxiv.org/html/2509.12741) | Uses real-world force observations and offline RL to adapt dressing | Our unreliable simulated force labels do not refute real force feedback; no reason to reopen force-map supervision now |

The proposed combination is a research candidate, not certified novelty. A
defensible contribution would require evidence that an explicit validation and
query-allocation rule turns imperfect IPC sensitivities into better policy
improvement per unit of computation, with isolated ablations and transfer beyond
dressing. Generic gradient mixing, trust regions, demonstration initialization,
and imitation of optimized actions are already well represented in prior work.

## 7. Implementation order and current status

Reuse `physics_gradient_probe.py` for snapshots and measured rollouts,
`physics_gradient_actor.py` for proposal directions, and
`physics_gradient_finetune.py` for frozen-policy fitting experiments. Add bounded
candidate comparison and accepted-target provenance before touching the main
trainer. Integrate an opt-in verified-correction path into the existing trainer
only after those checks; preserve dense/residual defaults and saved architecture.

The first deliverable should be a dressing report that distinguishes proposal
quality, policy-fitting quality, and final rollout quality. No new force channels,
world model, recurrent architecture, long adjoint chain, or 27-teacher dependency
is required for that first test. Extra tasks later must preserve the same learning
rule; task-specific rewards and observations remain explicit adaptations.

Completed in this research pass: source audit, existing-data metadata check,
primary-source comparison, and this falsifiable design. The earlier 40k study
was stopped before completion; its saved logs are interim diagnostic evidence,
not final 40k results. Native dressing verification and policy training for this
proposal have not run. No performance, robustness, or real-world gain is claimed.
