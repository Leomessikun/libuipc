# Replacement RL: architecture reset and a rejected novelty shortcut

Date: 2026-09-20. Branch: `research/joint-policy-query`.

## Decision and actual status

The owner explicitly removed two design requirements: compatibility with the
current SAC/pretraining structure, and an IPC-specific contribution. Neither is
a requirement for the replacement algorithm. Existing trajectories, completed
experiments and simulator implementations remain usable experimental resources.
No code/data deletion, simulator migration, process termination, or new long
training run follows from this scope change.

**This research pass does not establish a publishable new algorithm.** It develops
and checks a simulator-independent branching policy-gradient candidate, then
identifies substantial existing theory covering its core. A proposed extension
does not beat the simpler reference in the first synthetic check. Do not promote
this candidate as the selected replacement or a demonstrated dressing solution.
This finding must not be relabelled as a positive new-algorithm result later.

The requested deliverable remains a specific learning mechanism with a defensible
difference from close prior art and a matched policy-training experiment. Another
SAC loss, force channel, renamed search-and-imitation loop, or simulator-fidelity
argument does not satisfy that deliverable.

## Evidence that constrains an algorithm, without prescribing its architecture

- The [ordinary-SAC action audit](2026-09-20-action-selection-evidence.md)
  does not support declaring action gradients universally useless. Replacing
  the selector alone did not produce successful full continuations.
- The [matched recovery transfer](2026-09-20-recovery-transfer-audit.md)
  establishes successful feedback control and failed transfer on one development
  cell: teacher 8/8, each learned continuation 0/8. These are repeated development
  continuations, not independent tasks. All fail the separate historical paper
  filter. This cannot stand in for an explanation of ordinary SAC's failures.
- Candidate-ranking improvements that need thousands of simulated decisions per
  state target have not justified their workstation cost. A new method must
  improve the executed policy's complete episodes, including collection and
  evaluation time in its cost.
- The [inventory review](2026-09-20-training-issue-review.md) prevents using
  continuity of reward joins as proof of zero slope, or the confounded privileged
  run as a matched perfect-information upper bound.

## Close prior art read in this pass

| Proposed mechanism | Existing work and consequence |
|---|---|
| Generate successful controllers and fit a deployable policy | [MDGPS](https://arxiv.org/html/1607.04614) explicitly handles approximate policy projection; [PI2-GPS](https://arxiv.org/html/1610.00529) uses sampling for contact tasks. Search plus fitting is established. |
| Optimize around learner errors | [DART](https://proceedings.mlr.press/v78/laskey17a.html) estimates noise from learner error to collect corrective demonstrations. [To Distill or Decide, Section 6](https://arxiv.org/html/2510.03207v1) improves distillability by training smoother experts with motor noise. These are required comparisons, not novel mechanisms here. |
| Reuse promising prefixes rather than repeatedly start over | [Go-Explore](https://arxiv.org/abs/2004.12919) already develops return-then-explore and robustification. Resetting near success is not a contribution by itself. |
| Particle search inside RL | [SPO](https://papers.nips.cc/paper_files/paper/2024/file/01fb6de3360f9e32862665580e2c5853-Paper-Conference.pdf) uses SMC policy improvement and discusses optimistic selection of stochastic outcomes as a limitation. Its experiments use deterministic dynamics. |
| Direct history-policy updates from particle trajectories | [P3O](https://proceedings.neurips.cc/paper_files/paper/2025/file/3afe351e7b99b4b2f03c2ec9ce7ef2a8-Paper-Conference.pdf) uses nested SMC and a likelihood-score update for POMDPs. A memory policy plus particle sampling is already covered. |
| Correct probability mass after cloning trajectories | [Weighted ensemble](https://pmc.ncbi.nlm.nih.gov/articles/PMC2830257/) establishes statistical correctness for broad path processes. [Optimized weighted ensemble](https://pmc.ncbi.nlm.nih.gov/articles/PMC8378190/) already studies variance-driven allocation. Correction and allocation alone are insufficient novelty claims. |
| Emphasize rare successful trajectories | [Particle Value Functions](https://arxiv.org/html/1703.05820) explores particle objectives and rare rewards. Exponential return weighting is established and changes the objective in general. |

These comparisons are targeted method/limitation readings, not an exhaustive
claim to have read every related paper or ruled out every possible contribution.

## Concrete candidate: probability-corrected branching policy learning

This is specified to make the alternative technically reviewable. Its standard
components are acknowledged above. It is independent of SAC and differentiable
physics, but it is **not independent of policy-parameter gradients**.

Let a deployable recurrent policy be `pi_theta(a_t | h_t)`, with observation and
action history `h_t`. Let `Y(tau)` be the agreed terminal success indicator.
For dressing, success must retain the full grasp history and sustained coverage;
an arbitrary peak coverage is not this objective. Optimizing this indicator
instead of the old dense training return is an explicit objective choice. Every
matched baseline must receive the same objective.

The objective and score identity are

```text
J(theta) = E_pi[Y(tau)]
S(tau)   = sum_t grad_theta log pi_theta(a_t | h_t)
grad J   = E_pi[Y(tau) S(tau)].
```

Maintain N partial trajectories, each with a complete simulator snapshot,
observation history, policy memory, ancestry, and probability mass `w_i`.
Initially `w_i = 1/N`. Advance each with the frozen current policy and the
ordinary stochastic environment. At a scheduled branch boundary:

1. Use a positive priority to select parent i with probability `q_i`.
2. Sample N parent indices with replacement; restore each chosen history/state.
3. Assign each child of i the mass `w_i / (N q_i)`.
4. Continue with independent future action/environment draws conditional on the
   restored state. Preserve the full ancestral policy score.

At the terminal horizon, the update estimator is

```text
g_hat = sum_leaf w_leaf * Y_leaf * S_leaf.
```

One ascent step updates the actual executing policy. There is no intermediate
privileged teacher whose improvement then needs to survive distillation. Normal
policy backpropagation remains; action-value or simulator derivatives are absent.
This does not prove that eliminating either derivative is the source of a gain.

For any vector-valued future contribution `F_i`, one resampling step satisfies

```text
E[sum_child w_parent/(N q_parent) * F_parent | parents]
    = sum_i w_i * E[F_i | parent i].
```

Iterating conditional expectation gives the likelihood-score estimator above,
provided the snapshot/history is sufficient for the correct continuation law,
the proposals retain support, and the relevant moments exist. Priorities can be
inaccurate without biasing this identity; they can still make variance disastrous.
Total mass is conserved in expectation, not necessarily in each realization.
Self-normalizing, clipping weights, dropping ancestral actions, reusing the
tree for arbitrary off-policy epochs, or treating all cloned leaves as independent
would require new analysis. A zero vertex-position restore error alone does not
establish the required simulator continuation law.

This changes how an entire policy batch is collected. It does not require a
full candidate search at every training state. It also cannot create success
when no sampled behavior reaches it, resolve missing observations automatically,
or establish robustness to a new robot or garment from one successful branch.

## Proposed extension and why its novelty is unresolved

A possible branch priority uses the second moment of the **policy update**, not
only predicted success:

```text
M_i = E[ ||Y * (S_prefix,i + S_future)||^2 | history i ]
q_i proportional to w_i * sqrt(M_i)                 (equal-cost continuations).
```

For independent children and a fixed one-step resampling problem, minimizing
the conditional second moment `sum_i w_i^2 M_i/q_i` gives this allocation by
Cauchy--Schwarz. This is a local oracle result, not an optimal adaptive tree or
wall-time theorem. A learned approximation would need to account for the policy
changing, shared ancestry, future policy scores, and variable continuation cost.
The square-root allocation is standard importance-sampling mathematics.
Applying it to an augmented history containing policy scores may also be covered
by existing weighted-ensemble theory. **It is not a defensible novelty claim yet.**

There is another important limit to the apparent difference from P3O: for a
binary terminal indicator,

```text
log E[exp(eta Y)] / eta
    = log(1 + (exp(eta)-1) J) / eta.
```

For positive eta this is a strictly increasing function of J. Thus merely
switching a particle method to terminal success does not produce a new optimal
policy objective, or justify claiming that every risk-sensitive particle method
optimizes the wrong behavior in this particular binary setting. Finite-sample
estimators, conditioning on observations, and runtime still differ.

## Implemented CPU check and results

Reference script: `scripts/probe_weighted_branching_rl.py`. This is a standalone
NumPy mechanism probe, not a replacement neural learner or physics experiment.
The test has eight stochastic gates, zero intermediate rewards, terminal success
only, 32 particles and 10,000 independently generated trees per estimator.
Gate-action probabilities are known; **both priority functions are oracles**.
Branching occurs after gates 2, 4 and 6. Each tree consumes 256 counted decision
slots, including absorbing dead states. This is not a robotics wall-time match.

Exact success probability: **0.0035446542**.

| Estimator | Mean estimated success | Trace of gradient variance |
|---|---:|---:|
| Independent trajectory likelihood scores | .0031875000 | 1.88956e-4 |
| Corrected branching, success-probability priority | .0035520740 | 1.04587e-5 |
| Corrected branching, policy-score second-moment priority | .0036023709 | 1.05632e-5 |
| Same style of selection without mass correction | .3461375000 | 6.38300e-2 |

The simpler corrected reference reduces variance by about 18 times here. The
proposed score-aware extension does not improve it: its variance is about 1%
higher. Independent seeds are used across methods; this small difference does
not establish an ordering. Corrected gradient-coordinate means differ from the
analytic gradients by at most 1.18 and 2.73 estimated standard errors, respectively.
An exact enumeration of all two-child outcomes additionally checks conservation
of a signed, two-dimensional gradient, including duplicate ancestry. It exposes
the bias introduced by self-normalizing the child weights.

This is an intentionally favorable rare-event model with perfect priorities.
It establishes neither learned-priority efficiency nor a dressing improvement.
The uncorrected reference also preserves approximately the gradient direction
in this specially structured toy; its wrong magnitude/probability is not proof
that it chooses a worse learned policy. Importance-mass variability is recorded
and remains a concern. No end-to-end policy was trained.

Artifacts: `output/weighted_branching_rl_20260920/mechanism_with_mass_error.json`.
The earlier `mechanism.json` contains the same seeded experiment before adding
the final-mass standard error to reporting. The second execution is not new
independent evidence. Each four-arm script invocation took under one second on
CPU; no native physics or GPU job was started.

```bash
/home/ge47gax/kun/genesis-world/.venv/bin/python \
  scripts/probe_weighted_branching_rl.py \
  --out output/weighted_branching_rl_fresh/mechanism.json
```

## Research boundary after this pass

Do not restart the old dead-zone operator or advertise weighted branching as the
new paper. The proposed differentiating component has not demonstrated added
value, and the core has substantial prior art. The architecture reset is enacted;
the new-algorithm requirement is still open.

A publishable continuation needs one explicit claim beyond these references,
an implementation of that claim rather than a renamed baseline, and a same-budget
counterexample/experiment where the claim matters. Subsequent robot evaluation
must measure complete deployed-policy success and total workstation time on
multiple tasks, with identical observations, control periods and success rules.
The research should not be constrained to the existing simulator, optimizer,
pretraining interface, or a predetermined no-gradient explanation.
