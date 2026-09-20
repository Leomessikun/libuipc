# Research decision after the completed continuation and literature audit

Date: 2026-09-21. Branch `research/joint-policy-query`. This replaces the
interrupted staged synthesis. The requested literature screen and preregistered
experiment readout are complete; a novel RL algorithm and improved learned policy
have **not** been established. IPC and the current SAC architecture are optional.

## 1. Decision

Do not select event-boundary gradients as the main algorithm on the current
evidence. Keep active contact identification conditional on demonstrating an
information bottleneck. Defer real-world action-consequence calibration until
physical interaction data exist.

The strongest present empirical target is **learning a policy that makes progress
while preserving a trajectory-level validity condition**, under a strict query
budget. The useful distinction is between identifying an apparently good segment
and producing a deployable policy whose execution retains its benefit. This is a
problem statement, not a new algorithm name or a novelty claim. Standard
constrained RL, recovery policies, separate feasibility tests, and policy fitting
are essential baselines rather than contributions.

The recommended next research gate is therefore a small, matched **executable
constrained-policy improvement** test. It should establish whether a learned
policy can retain a measured feasible improvement and then complete the task.
This recommendation does not reinstate the old SAC/pretraining architecture or
restart the abandoned IPC-gradient line.

## 2. What the completed experiment adds

The [full result and reproduction command](2026-09-21-feasible-segment-result.md)
contain all 48 long branches, trace validation and the exact artifact hash.
One tshirt_26/body/checkpoint, eight slots and two repeats; decision 40 to 298;
eight intervention decisions followed by the same policy.

| Arm | Branch-valid grasp proxy | Sustained upper-arm coverage | Validity-weighted coverage | Valid success |
|---|---:|---:|---:|---:|
| incumbent policy | 0/16 | .557 | .000 | 0/16 |
| lift | 5/16 | .532 | .163 | 0/16 |
| outward | 16/16 | .468 | .468 | 0/16 |

Runtime was **18.53 minutes**: 12,384 branch decisions plus 320 approach decisions.
Outward demonstrates that very low immediate progress can precede substantial
valid late progress. It does not solve the task. Lift demonstrates the failure of
short-window certification: 16/16 pass at eight decisions, but 11/16 violate the
criterion later, at relative decisions 17–21. Both preregistered outcomes occur:
validity-weighted progress improves, and valid completion remains zero.

The collector omits the maximum tracking error of the approach prefix. These
positive counts therefore apply to **the continuation**, not the whole episode.
The zero-success result also holds under the stricter whole-episode criterion.
The proxy is anchor tracking within 2 cm, not verified physical grasp retention
or human safety. Slots/repeats are not independent garment/body draws.

## 3. Corrections to the previous research narrative

These corrections narrow inference while retaining the numerical observations:

- **Mixed repeated outcomes are not a located boundary.** The prior “jump”
  calculation is a within-state/macro, event-conditioned coverage difference.
  It does not estimate action-space one-sided limits, and outcome conditioning
  is not causal identification. Bimodality alone does not prove discontinuity.
- **The .18–.27 numbers are mixed-cell fractions.** They are not the probability
  that a repeated command flips its outcome. Three repeats also cannot certify
  deterministic behavior where no disagreement happened to occur.
- **The ~350x figure is not a lower bound.** It contrasts a single short label
  with an independent two-probability precision heuristic. Coupling, shared
  learning, score estimators and adaptive stopping can change the cost. The
  long experiment directly shows that the cheap positive label is insufficient.
- **Highest coverage is not proved highest training reward.** Branch traces do
  not store accumulated reward. The missing explicit grasp criterion is real;
  an assertion that there is no indirect useful training signal is too strong.
- **No full dressing success in the literature is false as a broad claim.**
  [Zhang and Demiris, Science Robotics 2022](https://pubmed.ncbi.nlm.nih.gov/35385294/)
  reports a learned-manipulation pipeline from garment preparation to manikin
  dressing with over 90% success. It is a modular pipeline, not necessarily a
  single end-to-end learned policy.
- **Cross-paper timing ratios are not controlled speedups.** Local training
  cost includes learning; published raw simulation steps use different tasks,
  hardware and time discretizations. SAPO's total runtime table cannot identify
  libuipc backward-pass cost.

The [corrected event analysis](2026-09-21-event-boundary-analysis.md) preserves
its preregistration and observations. Script fields retain historical names for
compatibility and now carry explicit interpretation metadata.

## 4. Novelty screen and ranking

The [literature audit](2026-09-21-research-literature-audit.md) supplies primary
links, reading locations, access limitations and the dressing capability map.

| Candidate | Direct prior-art boundary | Current evidence and disposition |
|---|---|---|
| Event-probability/boundary gradient | IPA/SPA/GLR, Lee et al. 2018, edge sampling, Teg, implicit-surface differentiation; AHAC/AGPO and mixed estimators | No measured action boundary or equal-cost gradient improvement here. Retain as a conditional estimator hypothesis; do not fund long training yet. |
| Decision-relevant active contact identification | Task-oriented sensing, haptic dressing classification, Deep Haptic MPC, POMDP-guided insertion | Early observation aliasing and feedback-specific value are unmeasured. A fixed outward direction already helps all sampled slots. Test passive history and privileged disagreement first. |
| Action-consequence simulator calibration | Value-aware learning/calibrated losses, real-data force models, prior garment sim-to-real calibration | No paired real interaction dataset. Defer the sim-to-real claim. Internal ranking validation remains useful engineering. |
| Cheap constraints plus expensive return queries | **Cai and Kandasamy, AAAI 2026: separate performance/feasibility tests**; optimal multi-fidelity BAI; CPO/Recovery RL/predictive filters | Most immediate local motivation, but the obvious allocator already has a direct precedent. A new contribution would need sequential learning and execution-error treatment with measured benefit. |

The fourth row is a refinement prompted by the data, not a reason to run a fourth
training stack. In particular, [the AAAI 2026 paper](https://ojs.aaai.org/index.php/AAAI/article/view/39063)
chooses whether to test performance or feasibility and eliminates arms using the
easier evidence. The general cost asymmetry is already part of its problem.
[Multi-fidelity BAI](https://arxiv.org/abs/2406.03033) further constrains a cost-based
novelty claim. Static identification is not sequential RL, but simply wrapping
that identification inside an actor update is not enough.

## 5. Formulate the objective before designing the estimator

Let `e_t` be the actual per-decision tracking maximum and
`S_T = product_t 1[e_t <= .02]`. Let `C_T` be the terminal sustained coverage.
Two objectives are useful but **different**:

- completion probability: `J_success(pi) = E_pi[S_T * 1(C_T >= .7)]`;
- progress under a chance constraint: maximize `E_pi[C_T]` subject to
  `P_pi(S_T = 0) <= delta`.

`E[S_T C_T]`, used descriptively above, is a third scalar objective; it does not
alone enforce an arbitrary failure tolerance. The experiment has improved that
scalar while leaving completion probability at zero in the sample.
A conventional constrained baseline can use the first-violation cost
`c_t = 1[e_t > .02] * product_{j<t} 1[e_j <= .02]`; then
`E[sum_t c_t] = P(S_T=0)`. History must retain whether a violation already occurred.
An unaugmented instantaneous classifier cannot represent this absorbing criterion.

Two immediate implications follow from the formulation, not new theory:

1. A positive short survival label only lower-bounds time survived; it does not
   prove future feasibility. For macro `u` followed by policy `pi`, the risk is
   `p_fail(s,u;pi)`, which changes when `pi` changes. A valid macro under one
   continuation need not stay valid after policy improvement.
2. For the absorbing objective `S_T C_T`, a realized failure makes the final
   score zero. Stopping failed branches at the first violation would reduce this
   run's branch decisions from 12,384 to 5,707, retrospectively. This known
   baseline needs measured batching-aware runtime before claiming cost savings.
   It is invalid for an objective that rewards later recovery or unmasked return.

A score-function estimate of a stochastic policy's success objective already
accounts for changes in event probability. A boundary correction is not an extra
reward bonus to add to it. Any proposed hybrid must define its sampling measure,
bias, variance and total computation against that baseline.

For **new RL research**, the unresolved question worth testing is whether query
allocation can target *improvements that survive execution by the updated policy*,
rather than only resolving the best privileged segment under the old policy.
This creates coupled uncertainty in continuation risk, attainable value and
policy projection. Existing GPS/SPO/P3O and constrained BAI must be shown
insufficient for a specific reason; that insufficiency is not established here.
A generic confidence score or additional classifier would not settle it.

## 6. Minimal next experiment and explicit decision gates

This is a proposed protocol, not an additional run started in this session.
Complete the endpoint/prefix logging audit first; freeze garment success geometry
and validity semantics before collecting confirmation data. Keep the existing
threshold for historical comparability and report corrected geometry separately.

Use independent, held-out garment/body/initial-state cells and report both
per-cell results and aggregate outcomes. The current t26 state is development
data. Match observation history, actuator limits, action-segment duration,
reset protocol, learning budget and evaluation horizon across compared methods.

| Gate | Smallest informative comparison | Decision |
|---|---|---|
| Is the gain simply objective alignment? | Existing learner with explicit first-violation constrained objective versus its original objective; same data and budget | If this captures the gain, report an objective/benchmark repair; do not claim a novel estimator. |
| Is there useful state-dependent choice? | Fixed outward macro versus held-out state-dependent constrained selection; privileged selection as an upper reference | If a constant direction explains the gain, a sophisticated information/query selector has no demonstrated value. |
| Can the deployed policy keep the improvement? | Measured intervention controller, policy fitted to its correction, and an executed policy update using matched labels | If projection loses the gain, prioritize observation/history/execution error over further search precision. Privileged summaries are not full-state upper bounds. |
| Does feedback add causal decision value? | Same physical probe with trained feedback-enabled and feedback-masked continuations; passive-history and privileged controls | Proceed with active identification only if feedback improves decisions beyond the probe's physical effect and available history. |
| Is a novel query rule worth its cost? | Uniform full continuation, first-violation stopping, constrained BAI-style allocation, and candidate rule | Require held-out improvement in executed validity/completion per total workstation hour, including all labels and fitting. |

Do not run every row as separate large-scale training. First establish a
constraint-aware, executable gain; use the resulting failure mode to choose the
next row. A bounded pilot can use a 3,600 s external cap with separately charged
evaluation. The point is a comparative result, not claiming that any particular
number of transitions suffices across tasks.

Stop the boundary-estimator line unless it measures a gradient cost/error
advantage on continuously parameterized reachable interventions. Stop the active
identification line if privileged observations do not reveal action disagreement,
or passive history removes it. Stop a new allocator claim if the known constrained
BAI baseline matches it. Continue to report partial progress separately from
complete valid episodes.

## 7. Workstation budget without misleading extrapolation

The observed local training average remains 16,469.5 / 270,000 = .0610 seconds
per transition, about **16.94 hours per million** if that pipeline and throughput
remain unchanged. Prior branch collections cost .142–.362 seconds per decision;
the completed long-continuation run averages .0875 because much more continuation
is amortized per restore. A single universal “query cost” is therefore inappropriate.

[FLASH](https://arxiv.org/html/2604.17513v1) and
[SAPO/Rewarped](https://arxiv.org/html/2412.12089v2) provide useful budget context,
but do not determine our simulator's attainable speed or our gradient overhead.
If a cheaper simulator enables the matched research test, using it is fully
consistent with the owner's request. It must still reproduce the validity and
control problem being studied; matching visuals alone is insufficient.

The literature screen, corrected evidence and completed experiment now support a
specific next test. They do not support announcing a novel RL method, declaring
first-order methods unsuitable, or claiming full dressing has been achieved.

## Verification completed in this pass

Both CPU analysis commands completed. All pre-existing numerical analysis outputs
match the original JSON exactly after removing the new interpretation metadata:
**2,424 branches across six collections**, correcting the earlier narrative count
of 2,400. The new 48-branch report independently recomputes trace metrics and the
5,707-decision stopping counterfactual. Deliberately incomplete, duplicated,
truncated, summary-corrupted and nonfinite inputs were all rejected. These checks
validate the analysis, not physical grasp semantics or policy generalization.
