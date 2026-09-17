# Research direction: allocating IPC computation to policy-gradient correction

Status: abandoned by the owner later on 2026-09-17. Retained as historical
research, not the next implementation plan. The owner now prioritizes
[ordinary SAC rollout diagnosis](2026-09-17-normal-sac-rollout-audit.md).

## Decision

The owner rejected SAC plus recovery imitation as the research contribution and
requested a deeper algorithm design. That objection is correct: the implemented
auxiliary MSE is a useful baseline, not an established novel algorithm. Its newly
launched comparison was stopped; no completed training comparison is available.

The recommendation is a narrower research question: **can selective, progressively
extended IPC continuations correct a policy-improvement estimate more accurately
per second than ordinary SAC or uniformly allocated simulation?** The proposed
contribution would be the joint allocation of query locations, horizon and
repetitions under expensive, variable-cost contact dynamics. Novelty is not
certified: every component below has precedents, and combining them alone is not
sufficient. A new allocation result and convincing experiments are needed.

## What the dressing evidence actually establishes

| Observation | Design implication | What it does not establish |
|---|---|---|
| IPC outward recovery succeeds in two independent verification executions | Useful finite motions exist for one failed configuration | General robustness, superiority to other simulators, or optimality |
| Recovery BC improves recorded-action MSE about tenfold but still fails the target sleeve | Evaluate closed-loop policy improvement, not imitation error | That every imitation method or SAC fails |
| Some branches reach near-full coverage and subsequently collapse | Short or peak-only labels can be misleading | That a particular fixed longer horizon always suffices |
| Earlier physical action corrections pass no state gates; full adjoint and repeatability are incomplete | Do not require an accurate full contact derivative as the first implementation dependency | That simulator derivatives are universally useless |
| Forward simulation dominated earlier dressing profiles; BC updates were fast | Optimize expensive simulation allocation, including construction and prefix replay | That high GPU utilization alone produces a better policy |

These observations do not isolate critic error as the sole cause of dressing
failure. Exploration support, observation aliasing, grasp/control behavior and
state restoration can each defeat the proposal. The first mechanism test must
establish that correcting the policy gradient is useful on this task.

## Closest primary sources and novelty boundaries

| Work | Existing mechanism | Boundary for our claim |
|---|---|---|
| [DAgger](https://proceedings.mlr.press/v15/ross11a.html), [Guided Policy Search](https://proceedings.mlr.press/v28/levine13.html) | Learner-state teaching and trajectory-guided policy learning | Recovery supervision and trajectory distillation are established |
| [Q-Prop, ICLR 2017](https://research.google/pubs/q-prop-sample-efficient-policy-gradient-with-an-off-policy-critic/) | Off-policy critic as a policy-gradient control variate | Correcting an actor gradient with sampled returns is established |
| [Stein action-dependent control variates, ICLR 2018](https://arxiv.org/abs/1710.11198) | General action-dependent baseline plus reparameterization correction | The basic gradient identity below is not ours |
| [The Mirage of Action-Dependent Baselines, ICML 2018](https://proceedings.mlr.press/v80/tucker18a.html) | Audits variance claims and implementation-induced bias | An unbiased identity is not evidence of a useful variance or policy gain |
| [TPX, ICML 2023](https://proceedings.mlr.press/v202/parmas23a.html) | Scalable mixture of likelihood-ratio and reparameterization gradients | Combining derivative-free and analytic gradients is established |
| [AHAC, ICML 2024](https://proceedings.mlr.press/v235/georgiev24a.html) | Adapts differentiable rollout horizon around stiff contact dynamics | Contact-aware horizon adaptation by itself is established |
| [Dynamic-horizon MVE](https://arxiv.org/abs/2009.09593) | Adapts model-based value expansion horizon | Adaptive return estimation is established |
| [Multi-Fidelity Policy Gradients, TMLR 2026](https://arxiv.org/html/2503.05696v4) | Correlated high/low-fidelity trajectories as a control variate | Mixing a cheap solver with IPC is not, by itself, new |
| [IADD-TR, August 2026 preprint](https://arxiv.org/abs/2608.10634) | Replay-policy-gradient targeted residual correction with dynamics factorization | Counterfactual or doubly robust correction is also a crowded claim |
| [Probabilistic Chunk Masking, May 2026 preprint](https://arxiv.org/abs/2605.16154) | Allocates VLA gradient computation by phase-level outcome divergence | Selectively spending learning computation is established; its reported bottleneck is different from ours |

Random-horizon gradient estimation also predates this proposal; see
[the SIAM paper on policy-gradient convergence](https://par.nsf.gov/servlets/purl/10218458).
The potential distinction is **joint allocation of expensive forward IPC queries
and continuation depth for a SAC improvement correction**, with explicit
selection-bias accounting and measured solver variation. The search did not
establish that this exact formulation is absent from all prior literature.

## A precise objective before an algorithm name

Freeze a reference policy $\bar\pi$, critic $q_\phi$, temperature $\alpha$ and a
distribution $\mu$ of complete simulator root states $x$. The actor observes
$o=O(x)$. Draw the initial action from the current reference policy; all branch
continuations use $\bar\pi$, not a geometric teacher.

For a branch of length $h$, define a soft return excluding the initial action's
entropy, consistent with SAC's Q convention:

$$
Y_h=r_0+\sum_{t=1}^{h-1}\gamma^t
[r_t-\alpha\log\bar\pi(a_t\mid o_t)]
+\gamma^h\bar V(o_h).
$$

At a genuine terminal state the bootstrap is zero. The current dressing trainer
treats step 300 as a time limit, not a Bellman terminal. Therefore a branch ending
there must retain the agreed bootstrap; removing it silently changes the
objective. Full continuations eliminate intermediate bootstrapping, but do not
make a terminal bootstrap exact. Use the existing scaled SAC reward consistently.

The immediate target is the gradient, at $\theta=\bar\theta$, of the fixed-root,
fixed-continuation surrogate
$\mathbb E_{x\sim\mu,a\sim\pi_\theta}[\mathbb E(Y_H\mid x,a)]
+\alpha\mathbb E_{x\sim\mu}\mathcal H(\pi_\theta(\cdot\mid O(x)))$.
It is not automatically the gradient of the full task's on-policy visitation
objective. Partial observations do not invalidate a baseline identity, but can
still prevent the unchanged reactive actor from representing a robust controller.

## Estimator and selection accounting

Choose nested horizons $0<h_1<\cdots<h_L=H$ and let $Y_0=q_\phi(o,a)$.
For a root/action sample, let $I_l$ indicate that the branch was actually run to
level $l$, and $P_l>0$ be its marginal probability of reaching that level,
including the probability of selecting the root for a query. Define

$$
C=\sum_{l=1}^{L}\frac{I_l}{P_l}(Y_{h_l}-Y_{h_{l-1}}).
$$

Choose each continuation probability before observing the outcome of that next
segment. Under valid conditional sampling and complete root restoration,
$\mathbb E[C\mid x,a]=\mathbb E[Y_H\mid x,a]-q_\phi(o,a)$.
This is telescoping plus inverse-probability weighting, not a new theorem.
Always retain positive probability of reaching the longest horizon. Deterministic
stopping at an apparently easy state would generally lose this property.

The proposed actor update uses the ordinary SAC path through the critic plus
the likelihood-ratio residual:

$$
\hat g=\nabla_\theta q_\phi(o,a_\theta)
+\operatorname{stopgrad}(C)\nabla_\theta\log\pi_\theta(a_{\mathrm{fixed}}\mid o)
+g_{\mathrm{entropy}}.
$$

Equivalently, add $-\operatorname{stopgrad}(C)\log\pi_\theta(a_{\mathrm{fixed}}|o)$
to the existing SAC actor loss, averaging over **all roots**, including unqueried
ones. The action in this log probability must be detached. The critic parameters
and reference return labels are frozen for this update. The root action is sampled
from $\bar\pi$; taking one update at that policy avoids silently treating stale
teacher actions as on-policy samples. Reusing branches for many actor updates
requires explicit off-policy correction and a separate analysis of continuation
policy staleness. Clipping weights, selecting only successful branches, using
freshly refitted control variates on the same targets, or dropping rejected query
outcomes changes the estimator and its claims.

This uses forward IPC outcomes, not an IPC contact Jacobian. A wrong critic can
remain a useful baseline if the sampled residual corrects its error; it need not
first learn an accurate contact derivative. However, rare-query weights and long
returns can cause large variance. If the policy has negligible probability of
useful recovery motions, this estimator alone does not create that exploration.
The existing 360 heuristic recovery rows cannot be relabeled as current-policy
return samples. Existing trajectories remain useful for initial training and
finding candidate root states; new, targeted branch queries are still needed.

## The part that must become a research contribution

Allocation should minimize policy-gradient mean-square error under a measured
wall-time budget, jointly over root selection, continuation probabilities and
repeat counts. Use past query batches to estimate residual second moments,
repeat variability and cost; freeze the allocator during the next estimation
batch. Keep an explicit random exploration probability for unqueried regions.
Privileged simulator diagnostics can inform this training-only allocator without
adding runtime inputs to the actor.

For the simpler independent, one-level query problem, a familiar constrained
optimization gives $p(z)\propto\sqrt{m(z)/c(z)}$, clipped to a positive supported
range, where $m(z)$ is the conditional squared norm of the residual gradient and
$c(z)$ is query cost. This allocation principle is not novel. For nested
continuations, level covariance, conditional stopping, startup cost and shared
batch convergence matter. Applying the one-level formula independently and
claiming joint optimality would be unjustified. Deriving a useful allocation rule
or defensible bound for that joint problem is the theoretical work still needed.

Use persistent GPU worlds and reuse prefixes where state fidelity is established.
Count prefix replay, snapshots, rebuilds, teacher search, validation, and any
allocator fitting in the cost. The observed .049 coverage spread under identical
restored actions is an unresolved engine/recovery issue: identical positions are
insufficient evidence of identical full dynamical state. Validate the conditional
query distribution or account for its bias before making unbiasedness claims.
Numerical variability is not equivalent to physical domain randomization.

## Implementation and falsification sequence

1. **Mathematical core, completed:** `policy_gradient_correction.py` implements
   nested return residual weighting and the score-loss term, isolated from SAC
   training. Seven tests pass, including exact selection enumeration, gradient
   detachment and a discontinuous one-dimensional reward with an intentionally
   wrong critic. These verify algebra and code only; no cloth-performance result.
2. **Next native implementation:** extend the existing branch/tape machinery to
   record full roots, policy version, actual sampled action/noise and log
   probability, rewards/entropy, terminal bootstrap, horizon probabilities,
   restore audit and execution time. Use policy continuations, retain failures,
   and never reuse the heuristic teacher's success filter for gradient samples.
3. **Mechanism test before training:** at a small fixed set of learner-visited
   dressing states, compare the critic direction, uniform full-branch correction,
   and selective correction against an independently sampled larger reference.
   Report directional agreement, bias/variance and total query time. If selective
   correction is not better per second, stop the proposed allocation mechanism.
   If the critic direction already agrees, its error is not the right bottleneck.
4. **Only after that passes:** integrate the correction into scheduled pretraining
   actor updates with one frozen-policy query batch per update. Keep ordinary TD
   critic learning, actor architecture and deployment code. Compare plain SAC,
   SAC plus verified imitation, uniform branch correction, and the allocation
   proposal at matched wall time and matched initialization/data. Test the fixed
   final checkpoints in complete dressing episodes across multiple training seeds
   and genuinely withheld bodies/garments. Success and valid grasp matter more
   than an estimator loss or transient sleeve coverage.
5. **Generality:** if dressing improves, test another deformable threading task
   and a rigid contact task with expensive accurate simulation. Include a cheap
   simulator or equal-cost query ablation before attributing gains specifically
   to IPC fidelity. Multi-task results do not substitute for dressing or real-robot
   validation.

Current status: a researched candidate with a tested mathematical primitive;
no completed new RL algorithm, native correction collector, learned allocator,
stronger dressing policy, or established novelty. The auxiliary imitation code
remains disabled by default as a comparison baseline. No training is running.
