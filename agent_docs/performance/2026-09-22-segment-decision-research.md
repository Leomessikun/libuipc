# Direct decisions from operation consequences: research assessment

Date: 2026-09-22. Branch: `research/expo-ft-dressing`.
Status: literature and existing-code review; no tests, simulation, training, or
new empirical analysis. This supersedes the old Stage 0–3 execution plan. It
specifies a possible task-solving architecture, not a validated new RL algorithm.

## Decision

Retire the prerequisite of training a unified visual SAC policy from scratch in
IPC. Retire the teacher/student, goal-prediction imitation and distillation route.
Keep historical data and objective/evaluation repairs. Neither SAC nor IPC is a
required component of the replacement method.

The replacement architecture worth developing on paper is **direct planning with
learned operation consequences**: learn what executable gripper operations do,
including whether they violate the criterion during execution, and use that model
to choose the next operation at deployment. The planner itself makes decisions;
there is no subsequent teacher-to-student projection. Neural training of a model
is different from collecting new from-scratch RL interaction, but neither was
started here.

The literature rules out claiming that action chunks, variable duration, failure
prediction, offline planning, or their generic combination are new. The narrower
research question is whether sparse branched interaction can support useful
**changes of continuation** without mislabeling an operation using the old
continuation's outcome. A specific estimator or representation with a defensible
cost/error advantage is still missing. Do not sell the architecture as that result.

## What the existing evidence establishes

The [completed continuation](2026-09-21-feasible-segment-result.md) is one
tshirt_26/body/checkpoint, not a general dressing benchmark:

| Eight-decision intervention, then the same incumbent | Valid through intervention | Valid through recorded continuation | Sustained coverage | Valid completion |
|---|---:|---:|---:|---:|
| policy | 2/16 | 0/16 | .557 | 0/16 |
| lift | 16/16 | 5/16 | .532 | 0/16 |
| outward | 16/16 | 16/16 | .468 | 0/16 |

Lift's 11 delayed violations occur at relative decisions 17–21, after the
intervention ends. This does not identify their cause. They could reflect damage
already induced by lift, the subsequent policy's choices, or their interaction.
It therefore establishes neither that lift is intrinsically infeasible nor that
a different continuation could rescue it. Similarly, outward is not a certified
route to completion: its sampled continuations do not reach the coverage target.

The approach prefix was not fully logged; these are branch-valid results only.
Repeated slots are not independent garment/body draws. The 2 cm criterion measures
soft-anchor tracking in this simulator, not physical finger separation or human
safety. The new formulation preserves that historical proxy explicitly rather
than promoting it into a general physical safety certificate.

The [earlier corrected synthesis](2026-09-21-research-direction-synthesis.md)
already notes that continuation risk changes with the policy. This is a useful
constraint on method design, not a newly discovered causal effect or novelty.

Consequently, the old Stage 2 rule "one long validation fails, set the candidate's
weight to zero" cannot mean permanent elimination across changed continuations.
It can reject that tested operation-plus-continuation for the defined objective.
It cannot reject every future policy that starts with the operation. High local
repeat agreement does not fix this mismatch, nor prove population determinism.

## Closest prior work and what it rules out

Primary sources were checked on 2026-09-22. The descriptions below are bounded
method comparisons, not reproductions or an exhaustive novelty proof.

| Work and source | Already covered | Consequence for this project |
|---|---|---|
| [Q-chunking, Li et al., current manuscript v4](https://arxiv.org/html/2507.07969v4), Sections 4–6 | TD learning directly over action sequences, coherent exploration from offline data and chunk-length backups. | Replacing single actions with segments is not an RL contribution. Its reported offline-to-online results do not establish our single-workstation cloth cost. |
| [Improving planning and MBRL with temporally-extended actions, Chatterjee and Khardon](https://pecey.github.io/MBRL-with-TEA/) (author page identifies NeurIPS 2025) | Joint action/duration planning; learned temporally extended transitions; bandit selection of duration ranges. | Choosing direction plus duration and predicting an endpoint is already a concrete method. |
| [MAC, Park et al., author project](https://kwanyoungpark.github.io/MAC/) | Offline action-chunk policy, critic and dynamics; flow-based behavior proposals and rejection sampling for long model rollouts. | Offline data plus chunk dynamics and candidate selection is occupied. Large-dataset results are not evidence that 2,424 branches suffice here. Project page read; full manuscript was not retrieved in this pass. |
| [D-MPC, Zhou et al.](https://arxiv.org/html/2410.05364), abstract and method; [TMLR manuscript](https://openreview.net/pdf/3f07f72c470d040e35bc709485724578a8408163.pdf) | Learned multi-step action proposals and dynamics used directly in online model-predictive control. | Deploying the planner rather than distilling it is an established alternative. It is not the same citation as dressing's Diff-MPC baseline. |
| [VINE, Park et al., December 2025 preprint](https://arxiv.org/html/2512.03913v1), Sections III–IV and Appendix E | Offline success/failure data, option-level reach–avoid values and feasibility-guided tree search. Its derivation includes history context and option transition kernels. | Failure-aware segment composition, history context and reach–avoid Bellman equations are not new. Its pretrained VLA/low-level skills are not proposed for this project. |
| [Bernoulli-Continuation Policy, August 2026 preprint](https://arxiv.org/abs/2608.03483) | Learns continue/replan decisions for a frozen VLA from trajectory outcomes, with a runtime tradeoff. | Adaptive interruption is not a standalone novelty claim. Abstract checked; no independent verification of its reported gains. |
| [Barrier-enhanced flow matching, July 2026 preprint](https://arxiv.org/html/2607.29569v1), Sections III–IV | Barrier constraints over whole action chunks; demonstrated robot constraints include kinematic barriers. | Whole-chunk safety is occupied. A kinematic barrier does not automatically encode delayed cloth-anchor violations. Its guarantees depend on its stated assumptions. |
| [Cai and Kandasamy, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39063); [multi-fidelity BAI](https://arxiv.org/abs/2406.03033) | Selecting separate performance/feasibility tests; cost-aware sampling across fidelities. | The old query-allocation mechanism is not the new core. A short continuation is not a valid low-fidelity estimate of a different continuation without a justified relation. |
| [MOReL, NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/f7efa4f864ae9b88d43527f4b14f750f-Abstract.html); [SPIBB author overview](https://www.microsoft.com/en-us/research/project/spibb/) | Pessimistic offline models and uncertainty-restricted policy improvement. | Treating unsupported actions cautiously is standard. Improvement over an unsafe behavior policy does not imply task safety. |
| [Reachability MCTS/BRTDP, Ashok et al.](https://arxiv.org/abs/1809.03299) | Selective search with bounds on reachability. | Propagating upper/lower values and refining uncertain branches is also established. |
| [Garment Diffusion Models, author institution record](https://spiral.imperial.ac.uk/entities/publication/a0176f43-5eb7-4530-93d8-35c0118514df) | Predicts future garment-opening point clouds conditioned on observations/actions for dressing control. | Learning only task-relevant garment geometry and applying MPC is not an empty dressing niche. Its insertion result is not our whole-episode objective. |

A search result for an anonymous OpenReview safe-learning submission
(`TyslRDlFqD`) was not accessible beyond a browser challenge. It is not used as
evidence. Recent preprints above are not assigned unverified conference status.

## A concrete replacement mechanism, with its standard parts identified

### Learning target

Let h_t denote the available observation/action history, elapsed time, the
already-violated flag, and sufficient information to evaluate the final coverage
window. A compressed representation of h_t must preserve relevant hidden contact
uncertainty; instantaneous coverage and tool position are not assumed sufficient.
The simulated tracking flag is available locally; obtaining its physical analogue
is a separate deployment requirement.

Let u be an executable gripper-command segment with duration d. For example,
piecewise tool translations/rotations within the actual actuator limits can
parameterize u. These are commands the robot can execute, not desired cloth
vertex motions. The old seven macros are a limited recorded subset, not a
required action vocabulary or a demonstrated complete control basis.

Learn the joint kernel

$$
K_{u,d}(dh',\ell\mid h)
=P(h_{t+d}\in dh',\;\ell_{t:t+d}=\ell\mid h_t=h,\;\operatorname{execute}(u,d)),
$$

where ell is one if the tracking criterion is violated at any point during the
segment. The within-segment violation and successor history must stay coupled:
an unconditional endpoint model multiplied by an independent survival classifier
can make up safe endpoints that the data do not support. Equivalent conditional
factorizations are possible. This joint model is a standard option-model object.

The kernel depends on the executed segment and starting history, but not on the
policy chosen *after* it. The eventual outcome does depend on that continuation:

$$
p_{\rm fail}^{\pi}(h,u,d)
=P(\ell=1\mid h,u,d)
+\int K_{u,d}(dh',0\mid h)\,p_{\rm fail}^{\pi}(h').
$$

This total-probability identity explains the old label problem. Re-estimating a
different continuation does not require calling a segment intrinsically good or
bad, but it does require information about that continuation's reachable states.
Standard model-based RL already has this separation; the equation is not novel.

### Deployment decision and exact task semantics

For the historical finite-horizon objective, define

$$
Y=\mathbf{1}[\text{no violation through }T]\,
  \mathbf{1}[\min_{j=T-11}^{T} C_j\geq0.7].
$$

C_j is upper-arm coverage. The aim is to maximize E[Y], not merely survive or
maximize peak coverage. With exact history kernels, the segment backup is

$$
V_t(h)=\max_{u,\;1\leq d\leq T-t}
           \int K_{u,d}(dh',0\mid h)\,V_{t+d}(h'),
$$

with V=0 after a prior violation and terminal V_T equal to the sustained-coverage
indicator. A target reached before T is not automatically absorbing success:
that would change the existing last-12-decisions scoring rule. Retain the window
history through segment boundaries. This is finite-horizon model-based planning,
not a proposed new Bellman equation.

At deployment, choose a physically executable segment, execute the portion used
in its prediction, observe the resulting history, and plan again. If execution
is interrupted early, evaluate that prefix and the changed continuation; do not
reuse the unexecuted full-segment prediction. No teacher labels or student policy
are needed. IPC would be one optional source of transition evidence, not the
online evaluator for every imagined candidate.

There is no established safe fallback in the present evidence. Holding still,
retreating, or reverting to the incumbent must not be presented as certified safe.
If the data cannot support a feasible plan, the method must report that limitation
rather than manufacture a guarantee.

### Why this could help, and why it could still fail

The potential savings come from reusing learned local consequences across many
candidate continuations and planning directly, instead of spending physics calls
on every candidate and then learning an imitation of the selected result. They
do not come from skipping required physical substeps during actual execution.
No local speedup or training-time estimate is established here.

This also differs from the historical cloth-motion representation pretraining:
there is no auxiliary prediction loss followed by the same SAC actor training.
The learned model is used directly to evaluate actions at deployment. A compact
model is desirable, but no particular low-dimensional contact representation has
been shown sufficient; the history-kernel formulation must not hide that problem.

A model could still exploit unsupported action sequences, confuse occluded
contact states, or become so conservative that it never moves. The sampled long
branches contain no successful completions, so a model of those outcomes alone
cannot supply positive evidence of a complete route. Learning dynamics from
unsuccessful data is possible; reliable extrapolation to success does not follow.

## Existing-data contract: what is reusable today

This is a source/metadata inspection, not a rerun of the experiments.

| Artifact | Available evidence | Cannot be assumed |
|---|---|---|
| `collect_decision_branches.py` outputs: `result.json`, `states.npz` | Starting observations/history and privileged summaries; per-step coverage/tracking metrics; command arrays saved for slot 0. | Full branch-end observation histories for all slots, all executed commands for all slots, or valid approach-prefix history. The collector does not save these. |
| `replay.py::save` | Observation, next observation, action, reward and terminal mask; optional privilege/sequence metadata. | Consecutive rows necessarily belong to the same episode or cell. Sequence IDs are optional, and older data have different cost/observation semantics. |
| Corrected Stage 0 control replay metadata at step 7800 | 117,000 rows, observation width 5385, six-dimensional actions. | A ready-to-use, cross-cell sequential safety dataset; metadata alone does not certify stream identity, event reconstruction or support. |
| `collect_rollouts.py` | Per-attempt summaries; full observation/action arrays only for kept trajectories. | Full failed trajectories are not saved by this collector merely because failure summaries exist. |

Thus “train a compositional world model from the 2,424 branches” was premature.
Those branches support outcome analysis under the logged continuation. Other
replays may supply transition learning, but they cannot be merged blindly or
claimed to cover unobserved intervention sequences. No dataset conversion or
fitting was performed here.

## What could count as new research

The defensible candidate question is:

> Under a fixed expensive-interaction budget, can a task-relevant model reuse
> partial operation evidence across changing continuations while accurately
> representing the remaining uncertainty about valid completion?

For this to become an algorithm contribution, at least one concrete advance is
needed beyond the table above: a justified contact-relevant abstraction, an
estimator that reuses branched/censored observations with a sharper cost/error
relation, or a sequential improvement rule whose advantage survives comparison
with the direct known adaptations. Adding another head, ensemble, risk penalty,
duration selector or confidence threshold is not such an advance by itself.

The following limits can be established without another experiment:

1. **Survived prefix is not a positive terminal label.** Its future is unknown.
   A simulator timeout is not an observed constraint violation either. Timeout
   depends on state/action difficulty, so statistical noninformative-censoring
   assumptions cannot be silently applied.
2. **Old-continuation failure does not identify an untried continuation.** Two
   dynamics models can agree on every logged action and disagree on an unobserved
   subsequent action. No estimator can distinguish them from those logs alone
   without extra structural assumptions or evidence.
3. **Model error compounds.** For a fixed candidate policy with at most N segment
   decisions, a uniform total-variation error epsilon on the joint kernels along
   its reachable histories bounds bounded terminal-value error by at most
   min(1, N epsilon), using the usual simulation-lemma argument. This is not a
   new theorem or a bound supplied by our data. Optimizing the policy requires
   coverage of the histories reached by that optimization, not only logged ones.
4. **Safety semantics matter.** A short segment-level risk budget is not an
   episode-level risk budget. Endpoint-only feasibility loses intermediate
   failures. A tracking proxy is not a physical grasp/safety claim.

These limits narrow a viable research claim; they do not disprove model-based
planning. This pass selects a concrete alternative architecture and rejects
overbroad novelty claims. It does not establish the narrower new mechanism.

## Disposition of the old stages

| Old stage | Current disposition |
|---|---|
| 0: from-scratch constrained SAC comparison | Retired as a prerequisite and current execution plan. Keep objective/evaluation repairs and historical results. No restart. |
| 1: teacher/student goal representation | Retired by the owner, including DAgger and relabeled goal-imitation variants. |
| 2: query allocation, then policy fitting | Retired as the selected architecture. Known allocation methods remain related work; there is no student-fitting step in the replacement design. |
| 3: generalization and cost | Keep as eventual evidence requirements, not an instruction to launch experiments. |

The completed work here is the primary-source overlap screen, a precise
continuation-aware decision formulation, and an honest data/claim boundary.
No claim of solved dressing, confirmed novelty, or a guaranteed short training
run follows. The owner's no-test/no-training instruction remains in force.
