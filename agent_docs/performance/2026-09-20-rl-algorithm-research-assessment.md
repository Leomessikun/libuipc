# Independent assessment of the replacement RL algorithm direction

Date: 2026-09-20. Branch: `research/flow-latent-steering`.
Status: literature assessment and candidate algorithm specification. No new
dressing policy, native rollout, or training result. The scalar mathematical
counterexample below was checked on CPU.

Subsequent evidence: the [fixed-Q comparison and terminal check](2026-09-20-action-selection-evidence.md)
completed the previously missing selector diagnostic. The
[matched recovery-transfer audit](2026-09-20-recovery-transfer-audit.md) also
reopens the already-successful September 17 closed-loop teacher. Read these
follow-ups before treating the hypotheses and proposed tests below as untested.

## Owner's objective and recommendation

The objective is a **new RL algorithm for efficient, robust dressing**, not another
encoder pretraining stage or an obligation to retain SAC/FQL. Existing trajectories,
IPC execution and deployment observations can be reused without preserving the
current learning rule. Engineering corrections in the
[inventory review](2026-09-20-training-issue-review.md) remain necessary experimental
controls; they are not a substitute for this algorithm research.

**Do not select gradient removal as the solution yet.** The owner explicitly
asked for independent research, not agreement with that hypothesis. The present
evidence does not isolate action gradients as the dominant failure. Treat finite,
temporally coherent forward-search policy iteration as one competing algorithm
candidate, specified below so it can be falsified. Do not claim that all
first-order RL is unsuitable or that removing gradients alone is a contribution.

The closest foundational reference is Levine and colleagues' **Path Integral
Guided Policy Search**. It explicitly replaces a smooth local optimizer for
discontinuous-contact tasks with sampling-based policy improvement, followed by
visuomotor policy learning. This validates the architectural alternative, while
preventing us from presenting the architecture itself as new.
[Paper, sections III–IV](https://arxiv.org/html/1610.00529).

## 0. Independent evidence ranking and the immediate research question

| Explanation | Evidence in this project | What would distinguish it |
|---|---|---|
| Inconsistent task/control contract makes comparisons misleading | Directly verified scoring differences, control-rate changes and angular-rate mismatch | Freeze the same physical task for every algorithm; this is an experimental control, not the proposed contribution. |
| Useful behavior is rarely encountered or credited at the right duration | Historical replay has very little near-success experience; four-decision effects separate better locally | Keep the update rule and compare normal-start versus valid restored-state learning, and matched physical action durations. |
| Learned Q ranks consequences incorrectly | Plausible; failed correction/search studies do not isolate it | Compare the same candidate bank ranked by Q versus independent forward outcomes. |
| Extracting an action by following Q's local derivative is the main bottleneck | Not established; no clean matched selector test, and CEM also failed to establish robust gains | Hold Q fixed and compare its local-gradient proposals with sampled maximization; verify both by identical independent rollouts. |
| Available observations cannot identify the needed behavior | Plausible; 35D privileged summaries are not full state and previous comparisons are confounded | Match histories, dynamics and learner, then test observation/history/privileged inputs. |

The second explanation is a higher-priority **research hypothesis** than a blanket
gradient failure, not a proven cause. Sparse useful coverage, control timescale,
critic extrapolation and partial observations can defeat both local gradients
and black-box search. Do not select a preferred explanation merely because a
single diagnostic is consistent with it.

There is also direct counterevidence to the broad claim about dressing itself:
Wang et al.'s *One Policy to Dress Them All* explicitly uses SAC, and its method
and ablations emphasize policy/Q representation and pose decomposition. It
demonstrates that gradient-based RL can learn dressing under its conditions.
It does **not** prove it will meet our IPC single-workstation budget.
[Original paper, IV-B and V-B](https://arxiv.org/html/2306.12372).

RFCL illustrates a separate mechanism: changing which states the learner
experiences through reverse/forward curricula can address difficult exploration
without attributing failure to the optimizer's derivative. Its robotics results
are not dressing evidence; its reset mechanism is a required diagnostic control,
not a new contribution we can rename.
[RFCL, method and experiments](https://arxiv.org/html/2405.03379).

The immediate research question is therefore: **which change produces genuine
policy improvement at equal physical conditions and workstation cost: better
experience/temporal credit, better outcome evaluation, or a different action
selection operator?** The answer determines the replacement algorithm's core.
We can develop a new learner without precommitting to either preserving SAC or
removing every action gradient.

## 1. Which gradient is being rejected?

| Quantity | What it assumes/estimates | Position for this project |
|---|---|---|
| Physics/pathwise derivative, through `x_next = f(x,a)` | Sensitivity along the simulated trajectory; numerical/contact branch choices matter | Do not require it for the replacement algorithm. Prior failed IPC corrections remain closed. |
| SAC action-value derivative, `dQ(h,a)/da` | Local sensitivity of a learned value function, not a derivative of the simulator | Bypass it in the proposed improvement operator; measure whether finite behavior comparisons are more useful. |
| Likelihood-ratio policy gradient, `E[return * grad_theta log pi]` | Samples of outcomes and a differentiable policy distribution | Does not require differentiable contact. Contact nonsmoothness does not invalidate every policy-gradient method. |
| Supervised gradient fitting observations to an improved action distribution | An ordinary statistical learning objective | Retain if useful; it need not differentiate physical return or predict a useful infinitesimal action direction. |

Standard SAC does **not** backpropagate through IPC. A bad IPC Jacobian cannot
directly explain an ordinary SAC result. Likewise, a correct local physics
derivative does not ensure a useful long-horizon control direction.

A simple counterexample to "no useful physical gradient means no useful policy
gradient": let `R(a)=1[a>0]` and `a~Normal(mu,sigma^2)`. The sampled reward has
zero derivative almost everywhere, yet

`J(mu)=NormalCDF(mu/sigma)` and
`dJ/dmu=NormalPDF(mu/sigma)/sigma > 0`.

For sigma=1, mu=-1: expected reward 0.158655, derivative 0.241971. At mu=0:
expected reward 0.5, derivative 0.398942. The derivative of an expectation and
the expectation of naive pathwise derivatives need not coincide across a
discontinuity. This is an illustrative scalar example, not a model of IPC.

Suh et al. analyze how stiffness and discontinuities can hurt first-order
simulation estimators. A 2026 re-examination finds that estimator variance and
implementation also matter, and reports successful gradient combinations on
its robotics benchmarks. Neither paper establishes a universal no-gradient
rule for dressing. The latter's method discussion was read; its experiments
were not reproduced here.
[Suh et al., ICML 2022](https://proceedings.mlr.press/v162/suh22b.html),
[Onoda et al., 2026](https://arxiv.org/html/2604.18161).

"Zeroth order" also does not always mean mathematically gradient-free:
finite differences and evolution strategies can estimate gradients of smoothed
objectives. A Newton or natural-gradient replacement still needs informative
local estimates; changing the optimizer's order does not create a missing skill.

## 2. What our measurements support

The previous audit establishes important configuration/metric confounds, so
the argument cannot simply be "SAC failed, therefore gradients failed".

There are nevertheless reasons to investigate another improvement operator:

- A coordinated finite recovery can in principle cross a contact configuration
  that a tiny one-step change does not. The plausible dressing pattern is a
  temporary retreat/reorientation followed by advancement; this is a hypothesis
  to test, not an observed explanation of every failed episode.
- The existing calibration finds 16/24 local positive differences above three
  pooled SD for a four-decision perturbation. It concerns one upper-arm cell and
  short-term reward, not complete dressing or early threading.
- The earlier derivative study shows that an accurate local derivative did not
  yield a faster learner on the separate cloth-drag benchmark. This separates
  derivative correctness from end-to-end learning value.
- The **existing derivative-free CEM experiment also failed to establish robust
  improvement**. Its independent continuation evaluation produced zero sustained
  successes for every control. Mean coverage was 0.57704 for CEM+SAC versus
  0.57259 for SAC, and 0.58330 for the fixed reference. These were partial
  continuations, not full episodes.

Sources: [inventory review](2026-09-20-training-issue-review.md),
[CEM experiment](2026-09-17-parallel-dressing-trajopt.md),
[derivative benchmark](2026-09-14-iaql-benchmark.md).

Thus the research obstacle is not only finding a good sampled motion. It is
making a selected improvement survive independent simulation, realistic hidden
state variation, execution by an observation-conditioned policy, and the rest
of the episode. Removing derivatives does not remove those requirements.

## 3. Prior art and what cannot be claimed as new

| Work | Relevant established idea | Reading scope / consequence |
|---|---|---|
| [MPO, 2018](https://arxiv.org/html/1806.06920) | Nonparametric action reweighting followed by supervised policy fitting; no action-value gradient needed in this variant | Sections 3 and algorithm inspected. A decisive baseline for testing whether the critic's action gradient is the problem. |
| [PI2, 2010](https://proceedings.mlr.press/v9/theodorou10a.html) | Sampling and return-weighted policy improvement | Primary abstract plus the PI2 derivation/use in PI2-GPS inspected. Sampling-based RL is not new. |
| [PI2-GPS, 2017](https://arxiv.org/html/1610.00529) | Local sampling optimizer for discontinuous contact, coupled to a global visual policy | Main method read. Closest architecture; learner/teacher distribution matching is already part of GPS. |
| [RSPI, 2008](https://arxiv.org/html/0805.2027) | Rollout comparisons, significance-based labels and budget allocation | Main method reviewed in the preceding audit. Repeated branches plus supervised winners do not establish novelty. |
| [Robust MPO, 2019](https://arxiv.org/abs/1906.07516) | Robust/soft-robust objectives under transition-model misspecification | Abstract checked. Adding robustness to MPO is not by itself new. |
| [iCEM, 2020](https://arxiv.org/abs/2008.06389) | More sample-efficient trajectory search | Abstract checked. A stronger search baseline than a naive random candidate bank. |
| [Parameter versus action exploration, 2019](https://proceedings.mlr.press/v89/vemula19a.html) | Different dimension/horizon costs of exploration | Abstract and theoretical setup inspected. Entire-network ES is not automatically cheaper on one workstation. |
| [Multi-fidelity RL, 2014](https://thomasjwalsh.net/pub/icra2014Car.pdf) | Allocate experience across simulators of different fidelity/cost | Abstract/framework inspected. "Use cheap solves first" is not new. |
| [Adaptive action duration, 2025](https://arxiv.org/abs/2507.00030) | Learn action duration using a contextual-bandit mechanism | Abstract checked. Adaptive duration alone is not a contribution. |

This is a targeted primary-source review, not an exhaustive proof of novelty.
In particular, simply composing MPO, CEM, history and a confidence threshold
would repeat the problem identified with ADR 0009.

## 4. One candidate replacement loop, conditional on the selector/evaluator tests

The following specifies a competing algorithm family, not the selected solution
or a validated new method. Its basic policy-iteration backbone is established.
If ordinary gradients work after controlling experience and timing, drop the
claim that this architecture is necessary and identify what the experiments
actually support.

### A. Policy and decision representation

Let `h_t` contain deployment-available observation/action history. The policy
proposes a distribution over bounded motion segments, parameterized initially
by a few translation/rotation knots and a physical duration. It may alternatively
choose a short feedback program whose inputs are also deployment-available.
Use actual robot speed limits, not inconsistent per-decision limits.

This representation allows finite changes such as retreat-then-advance. It is
not restricted to a frozen flow prior's modes. Existing trajectories initialize
the proposal distribution if useful; they are not a permanent support boundary.
Include proposals outside that initializer and the current policy's behavior.
Do not search millions of perception-network parameters in the inner loop.

A plan evaluated as an H-step commitment must be executed as that commitment;
if it is executed with feedback/replanning, evaluate that same feedback rule.
Do not score one open-loop plan and silently deploy a different controller.

### B. Forward evaluation against the incumbent

Freeze policy version `pi_k`. At its visited simulator states, evaluate the
incumbent and candidate segments, then continue with `pi_k`. Record the
candidate-minus-incumbent return, whole-episode grasp status, verified completion,
solver setting and elapsed cost. The underlying task objective remains fixed
across these comparisons. No physics or Q action derivative is needed.

Short rollout scores may screen proposals; they are not certificates of full
task improvement. Continue finalists to a valid terminal/verification endpoint.
A learned tail value is optional but introduces another estimation error; omit
it from the first small mechanism test rather than claiming a critic-free method
while hiding a bootstrap. The score must permit temporary progress loss when
it produces better final completion.

### C. Improve a distribution, not one regressed winner

On a finite candidate bank with reference masses `p_i`, one conventional update
is

`q* = argmax_q [sum_i q_i L_i - eta * KL(q || p)]`,

where `L_i` estimates improvement over the incumbent. Its solution is
`q_i proportional to p_i * exp(L_i / eta)`.

This is a standard relative-entropy improvement step, **not our contribution**.
Keep the incumbent candidate and multiple good alternatives. The reference
masses must be specified: if candidates are sampled from a proposal different
from `pi_k`, this finite-bank KL is not automatically a trust region around
the old neural policy, and sample/proposal weighting cannot be ignored.

Fit an observation-history policy to the improved distribution, using a
multimodal likelihood or categorical action-program choices where appropriate.
Do not average two incompatible successful trajectories into an untested mean
trajectory. A neural supervised loss can use backpropagation while the control
improvement operator remains free of action/physics gradients.

### D. Close the RL loop

Execute the **fitted policy itself**, validate its improvement, collect its new
visited states, and repeat the search/update process. This is iterative RL based
on task outcomes, not cloning a permanently fixed teacher or running a planner
only at deployment. A successful planner is not yet a successful learned policy.
Full IPC state is training-only; the deployed controller gets `h_t` and requires
no IPC rollout. Real-world robustness remains a later empirical question.

## 5. Conditional contribution hypothesis: how to buy a reliable improvement

For that candidate, the narrow research hypothesis is a **joint policy-improvement and query-allocation
operator**: choose finite behavior duration, solver accuracy and repeats according
to which uncertainty prevents accepting an improvement over the incumbent.
Avoid a generic "contact-aware" label with no operational definition.

For a fixed candidate and continuation, distinguish:

1. **Sampling uncertainty**: repeat variability in the estimated mean improvement.
   More justified independent samples can reduce it.
2. **Numerical discrepancy**: the candidate-versus-incumbent comparison changes
   between solver settings. Repeating the same loose solve does not remove bias.
3. **Insufficient action duration**: the behavior has not had time to produce a
   meaningful physical effect. More repeats cannot make a true zero effect useful.
4. **Continuation/projection error**: the short score or fitted policy fails to
   retain the searched improvement. More accurate evaluation of the old search
   trajectory does not fix the new policy.

For a *fixed full-return comparison*, a possible conservative score is

`L_i = estimated_gain_i - sampling_radius_i - numerical_error_allowance_i`.

This is only a valid lower bound if the sampling assumptions and numerical
allowance are justified. A difference between two tolerances is **not** by itself
an error bound against exact physics. Without a defensible bound, call it an
empirically calibrated screening score, use a tighter reference on finalists,
and report actual validation error. Do not manufacture a guarantee from pooled SD.

The allocation decisions would be: repeat an uncertain comparison, refine a
numerically unstable one, propose a longer/different finite behavior when its
effect is negligible, or stop and retain the incumbent when no useful gain can
be resolved within budget. Changing duration changes the candidate, not merely
the measurement fidelity. Every duration pays the same physical-time task cost.

An implementable starting scheduler is to sample each candidate with a small
fixed budget, use cheap results for ordering only, refine contenders against a
common reference, and independently validate the selected distribution after
policy fitting. Learn/adapt the allocation rule only after measuring which error
source actually dominates. Do not irrevocably prune candidates using unbounded
low-fidelity error or assume returning the incumbent always satisfies a safety
constraint. The common episode scorer remains authoritative.

**Why this could be interesting:** contact tasks may make fixed sampling budgets,
fixed action durations and fixed solver accuracy inefficient in different phases.
The desired result is a better cost/decision-error tradeoff that translates to
policy improvement. **Why it might not be novel:** multi-fidelity optimization,
adaptive duration, robust policy iteration and GPS projection controls already
cover its components. A new joint rule, a defensible analysis or a distinct
demonstrated mechanism is needed; their assembly alone is insufficient.

## 6. Experiments that distinguish the hypotheses

**Execution update:** the owner's reminder led to a
[completed-test map and saved-data reanalysis](2026-09-20-action-selection-evidence.md).
Local gradient/finite-difference corrections, CEM, macro intervention, full
recovery episodes and duration calibration already have results. Reuse them;
the missing fixed-Q selector comparison and an independent full-continuation
check are now complete. Sampled-Q selection did not establish an advantage over
gradient extraction; one repeatable local gradient benefit did not translate to
full dressing on fresh executions. The proposed reset control also has a partial-start Newton reference
attempt, which did not demonstrate full-start success.

Do not launch another full-grid training run to test the entire idea at once.
Reuse the existing data, reset facilities and scorer corrections.

**Experiment 1: does nonlocal behavior search solve a problem local improvement
does not?** Use saved learner-visited states spanning threading, forearm stalls
and actual elbow contact, rather than only the existing already-on-upper-arm
calibration. Predeclare a small development set and a one-hour initial native
budget. Compare the incumbent, learned-Q-gradient proposals, matched random
finite segments and sampling-optimized segments. No new IPC adjoint work.
Control physical movement budget and measure independent complete continuations
for finalists. A larger transient reward is not a pass. If no useful finite
intervention exists in this budget, do not train on noisy winner labels.

**Experiment 2: isolate the optimizer and value estimator.** Separate
gradient-based versus sampled action selection from learned-Q scoring versus
forward-rollout scoring. In particular, sampled-Q selection tests the action
gradient hypothesis; sampled-rollout selection also changes the evaluator and
cannot attribute its gain solely to removing gradients. Use the same candidate
representation and charge forward-query costs. This experiment decides whether
the bottleneck is local search, critic values, or both.

Run a complementary short control with the **same gradient learner** and
physically valid restarts near demonstrated progress states. Compare against
normal starts and against matched action-duration controls. Success only from
these restarts implicates reaching/crediting useful experience; it does not
certify a full-task policy. Replay observation vectors alone cannot implement
this reset: use complete simulator snapshots or charged prefix regeneration.
Evaluation must ultimately return to the original initial-state distribution.

**Experiment 3: can the improvement be learned and retained?** On the same
initialization and data, compare fixed-budget rollout policy iteration / PI2-GPS,
a suitable MPO implementation, and the candidate joint allocation operator.
Use iCEM as a search control and the existing SAC as a reference. Test the policy
after distribution fitting, then after multiple improvement rounds. Include a
fixed-best-program control, multiple independent training seeds for any final
claim, and held-out garment/body cases. Same initial grid is not generalization.

The principal metric is whole-episode verified dressing improvement per total
workstation hour, including setup, search, fitting and evaluation. Log query count,
physical frames, repeat cost, numerical-reference agreement and failed grasps.
If adaptive allocation does not beat the same fixed-budget operator, drop that
claimed contribution even if the derivative-free baseline works.

For a general RL claim, later add other contact tasks with different failure
structure, without redesigning the operator per benchmark. Do not use the
already-saturated cloth-drag benchmark as the only evidence.

## Scope of the conclusion

The supported position is: **the current evidence does not identify removing
action gradients as the real solution**. The unsupported position is:
"first-order RL cannot solve dressing." Reopening the learning structure is
appropriate, but the research must discriminate mechanisms before selecting an
operator. The forward-search loop is a concrete, falsifiable candidate; it is
not another SAC pretraining patch, an established contribution, or a reason to
ignore evidence favoring a different replacement design.
