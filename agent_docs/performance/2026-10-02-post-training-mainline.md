# Dressing post-training: evidence, prior art, and the continuation decision

## Dynamic baseline status

October 3 update: the common-candidate privileged causal teacher completes 4/4
valid runs, current-pose planning 2/4 (both lift), and GICP 0/4 valid runs plus
one separately retained physics-invalid attempt. The single same-settings retry
was physically valid and lost the grasp at decision 166. The feasibility gate
passed; all eight training-motion preparations finished, and the first two
teacher initialization episodes are running. See the
[M4 results](2026-10-02-observed-motion-student.md#m4-validation-results--2026-10-03).
No dynamic student is trained yet. These results support bounded distillation
on the pilot, not a new algorithm or generalization claim.

M3's observed-motion teacher succeeds on pass but fails lift; current-pose
planning succeeds lift under common candidates. Its independent-reset check
then errors on a 0.136 mm cloth difference. No research student was trained.
The [M4 baseline](2026-10-02-observed-motion-student.md#m4-privileged-causal-teacher-baseline)
uses privileged recent-state causal forecasts only in the training teacher,
two validation/evaluation repeats and a held-out passive-motion preflight.
Students still use observable history. M4 has revalidated the teacher with common
candidates; legacy 6/6 is not pooled with this result. This supplies a bounded
DAgger control and possible initialization for the post-training learner below, without
changing the selected algorithm question or claiming a new teacher/student
method. EXPO and the shared consequence metric are not implemented yet. The
literature audit below downgrades the metric proposal's novelty and priority.

## Research assessment and bounded continuation — 2026-10-03

**Recommendation:** finish the already bounded M4 student experiment, but do not
scale collection or describe the current implementation as a new post-training
algorithm. The project has a useful simulator and an encouraging teacher pilot;
it does not yet have a demonstrated dynamic student, joint garment/motion
generalization, or a defensible new policy-improvement operator. This assessment
does not stop or reconfigure the running queue.

### What is actually running

The teacher tries 12 candidate actions with H=4 held-action continuations
(0.4 seconds). Its causal forecast extrapolates recent **privileged body states**;
it does not read the actual future GRAB sequence. The selected action becomes a
supervised target. The student adds a small four-frame residual MLP to frozen
r1 features/actor logits, using action MSE plus a residual penalty. DAgger adds
labels at student-visited states. There is no critic, RL editor, learned human
forecast head, or diffusion/flow learner in this pipeline.

Each of the four successful teacher runs takes about 2.2-2.9 hours for 45 seconds
of simulated execution; individual planning queries take about 146-189 seconds
on average. Distillation therefore has a concrete computational purpose. It is
not already a novel algorithm merely because it removes this online cost.

In all four runs the first selected action differing from r1 occurs at 1.1 or
1.2 seconds, after the 1.0-second motion onset. The human motion lasts only the
early part of the episode, and planning is restricted to elapsed time 0-5 s.
These runs support improved response to motion; they do not demonstrate
pre-onset anticipation or continuous dressing under sustained human motion.
The current-pose planner already succeeds on both lift repeats. Two clips on
one body cannot establish generalization or a population-level success rate.

### Primary-source prior-art audit

The search followed overlaps from privileged imitation to dressing MPC and then
to post-training objectives. It is not a proof of exhaustive coverage or absence
of prior art. Recent arXiv work is treated as a preprint, not verified external
replication. The most consequential overlaps are:

| Proposed claim | Closest prior art | Consequence for our claim |
|---|---|---|
| Privileged future-aware planner teaches a history point-cloud policy | [GenH2R, CVPR 2024](https://arxiv.org/html/2401.00929v2): privileged demonstration planning, historical point-cloud imitation and auxiliary future prediction for handover | The teacher/student recipe is established, although handover differs from cloth dressing |
| Human prediction plus MPC enables dressing | [Synchronous dressing support, 2026](https://link.springer.com/article/10.1186/s40648-026-00350-9): predicted human/force states in MPC | Prediction plus dressing is not new. Its hemiplegia-inspired protocol keeps the assisted arm still and differs from our moving-recipient-arm setup |
| Predict cloth consequences and improve a dressing controller | [Garment diffusion models, RA-L 2025](https://spiral.imperial.ac.uk/entities/publication/a0176f43-5eb7-4530-93d8-35c0118514df): action-conditioned garment-opening dynamics, MPC and iterative learning | Diffusion here models dynamics; adding prediction/optimization to dressing is already explored. This entry is based on the official abstract |
| Static dressing policy plus registration is the complete competitor | [Dressing in Motion, 2026](https://arxiv.org/html/2609.04759v1): diffusion policy, PDE-based region representation and hierarchical motion adaptation | Our r1+GICP is a registration proxy, not a full reproduction of Sun et al.; beating it does not establish beating that paper |
| Learn corrections, select with value, absorb into a pretrained policy | [EXPO](https://arxiv.org/html/2507.07986v3), [EXPO-FT](https://arxiv.org/html/2605.25477v2) | This is an existing reference method, not our algorithm contribution |
| Query consequences at key states, compare candidates, post-train the policy | [WISE, 2026 preprint](https://arxiv.org/html/2609.03681v1): scheduled bounded imagination, repeated candidate ranking and relative policy feedback | Replacing a learned world model with IPC is insufficient by itself; even selective querying and repeat-based filtering overlap |
| Weight imitation by the consequences of action error | [Levine et al., JMLR 2016, section 4.3](https://www.jmlr.org/papers/volume17/15-522/15-522.pdf): teacher-precision-weighted supervised updates; [MPC Q-loss, CDC 2023](https://publications.syscop.de/Ghezzi2023b.pdf): exact cost-to-go loss and a Gauss-Newton approximation | The earlier metric proposal has stronger prior art than the initial review recorded; generic consequence weighting and inverse/direct precision coupling cannot carry novelty |
| Adapt privileged supervision to what the student can observe | [A2D, ICML 2021](https://proceedings.mlr.press/v139/warrington21a.html), [GPO](https://arxiv.org/html/2505.15418v1) | An information gap is a plausible diagnostic, but teacher/student alignment is itself an established research problem |

Adjacent overlap also matters: [CRAFT](https://arxiv.org/html/2605.04470v1)
combines counterfactual advantages with grounded interactive corrections;
[CritiQ/ReTRy](https://arxiv.org/html/2505.09546v1) address difficult privileged
distillation with selective queries and recovery resets;
[CLIC](https://arxiv.org/html/2502.07645v3) uses action-set supervision. None is
the same dressing system, but their ingredients must not be repackaged as new.

### What would justify further research

The concrete question remains: **which training signal turns limited expensive
physical interaction into a deployable policy improvement on unseen garments
and human motions, beyond standard corrective imitation and established
post-training methods?** Success per total simulation/learning hour is an
endpoint, not an algorithm. A new mechanism still has to be identified and
compared with its closest existing solution.

The next three experiments should answer distinct questions, in this order:

1. **Finish M4, without expanding its budget.** Eight teacher initialization
   episodes (including the held-out validation body), matched history/current
   students with three training seeds, four shared DAgger roll-ins, then the
   existing held-out body/sequence evaluation. Report every task cell and seed,
   sustained completion, grasp failures, invalid physics, runtime and cost.
   This tests whether the teacher's improvement transfers to an observable
   policy. It does not test ClothesNet generalization or establish anticipation.
2. **Check simpler explanations before calling it predictive control.** On a
   small fixed held-out set, compare the best student with a bounded observable
   follow/pause controller and a fixed early-correction controller. Vary motion
   onset and speed, use matched schedules and two repeats, and retain the
   current-pose planner comparison. If simple early corrections explain the
   gain, use that finding and narrow the claim. This is a proposed follow-up,
   not a job launched in this review.
3. **Run one bounded mechanism test only if a consequential gap remains.** For
   12-24 recoverable teacher/student states, compare base, teacher and student
   actions using the same human motion and a fixed continuation policy after
   the first action. Repeat a subset to separate ranking noise from effects.
   Relate action error to downstream progress/grasp outcomes. If the student
   cannot fit decisions, check frozen-feature information/capacity with a
   privileged-input diagnostic and an observable geometry/history alternative
   on the same labels before attributing failure to the learning objective.
   Privileged input is diagnostic, not deployable. Existing endpoint-only
   branch logs cannot substitute for complete saved states and continuations;
   any added replay/collection cost must be counted. If cost-sensitive fitting
   is indicated, ordinary MPC Q-loss/GPS-style weighting is the first control,
   not a new method to rename.

Predeclare a practically meaningful gain before further scaling (for example,
10 percentage points of held-out completion at a fixed total budget), and size
the subsequent task set for that question. The four-cell M4 holdout is a screen,
not sufficient statistical power for that target. Analyze at task-cell level,
accounting for shared bodies, motion sequences and training seeds; candidate
branches do not create new tasks. Report uncertainty rather than interpreting
a nonsignificant result as equivalence.

**Continue** if observable students show a reproducible, practically useful
gain and the remaining failure has a testable mechanism beyond simpler controls.
**Stop scaling the current teacher/adapter route** if the fixed pilot and one
targeted diagnostic fail to produce such a gain. A frozen-feature adapter's
failure does not prove all dressing learning impossible. If established methods
already solve the problem, pursue a clearly scoped systems/empirical dressing
contribution or stop the algorithm-novelty claim; do not keep collecting until
a familiar method can be relabeled as new. Hardware transfer and both garment
and motion holdouts remain untested, rather than promised consequences.

## Scope correction

The owner clarified that the research remains **post-training an existing
dressing policy for moving arms and unseen garments**, using the recipe in
[Towards Universal Post-Training for Robotics](https://pd-perry.github.io/posts/post-training.html)
as a reference. The contact-friction intervention proposal is not the selected
next direction. Do not start its mechanism experiments as a prerequisite.

The concrete reference learner is EXPO-style offline-to-online policy improvement:
a pretrained base, a small learned action editor, a critic, online replay, and
supervised updates that absorb improved behavior into the base. Existing dynamic
DAgger remains a necessary baseline and source of initialization data. It is not
already an implementation of this RL loop.

## What the linked work actually supplies

[EXPO](https://arxiv.org/html/2507.07986v3) trains a small editor against Q while
training the base with an imitation objective. It selects among original and
edited actions for behavior and TD targets. The interface is not restricted to
VLAs. [EXPO-FT](https://arxiv.org/html/2605.25477v2) adds chunk-level learning and
human interventions. These are established methods to implement faithfully as
references, not new contributions here.

The current [Real-Time EXPO-FT paper](https://arxiv.org/html/2609.18207v1),
appendix VII-E, explicitly updates the base with flow-matching BC, including
LoRA and vision/action parameters. The older local note saying that its base is
entirely frozen is incorrect for this paper. Its latency handling is also not
our proposed novelty. A small bounded action edit is not a safety guarantee in
contact-rich dressing.

## Existing code and data audit

- `train_dynamic_student.py` regresses selected teacher actions with a trust
  penalty. It has no TD critic, return-maximizing editor, or online RL update.
- At inspection, observed/pass had 20 queries and 240 candidate results;
  observed/lift had 18 queries and 216 candidates. Fourteen candidate branches
  reported simulation errors. These are partial teacher logs, not 456 independent
  trajectories, verified task improvements, or completed student results.
- Candidate logs include actions, endpoint progress, attachment tracking and
  feasibility. They do not contain each branch's successor observation or a full
  reward/terminal sequence. Completed M3 trajectory archives also lack explicit
  reward and termination arrays. They are usable for imitation and short-horizon
  diagnostics, not plug-and-play Bellman replay.
- Current candidates hold the same action for H=4 decisions, while actual control
  replans after one. Their scores must not be mislabeled as one-action Q targets.
- `train_flow_policy.py` uses a four-dimensional action convention and a force
  feature; the dynamic adapter uses six actions and geometry/history. Reusing a
  checkpoint requires an explicit observation/action compatibility audit.
- Prior September results used other policies/protocols. They motivate checking
  edit support and critic ranking; they do not rule out EXPO on the current r1.

## Selected baseline implementation route

1. Start with current r1 as the pretrained base in its existing action interface.
   EXPO does not require converting it into a large diffusion model first.
   Use causal point-cloud history and measured proprioception for the editor
   and critic. Retain the existing flow checkpoint as a later expressive-policy
   comparison once its interface and geometry-only initialization are matched.
2. Initialize replay with valid existing transitions where outcomes can be
   reconstructed reliably. Store new full transitions explicitly. Teacher
   corrections may initialize the editor; all autonomous successes and failures
   train the critic. Invalid-physics branches are not task failure labels.
3. Learn a bounded stochastic editor, sample original and edited proposals, and
   select using twin critics. Tune edit scale on validation tasks; do not assume
   the successful behavior lies within an arbitrary small radius.
4. Execute, append replay, update critics and editor, and update the base with
   the reference method's supervised behavior objective. Track base-only success
   as well as base-plus-editor success to test whether improvement is absorbed.
5. First preserve one-decision execution. If adopting action chunks, use the
   same executed duration and terminal semantics in all comparisons. A chunk
   ending at H uses the corresponding discounted return and gamma-to-H backup.

Use one shared task objective and endpoint across methods: sustained sleeve
completion with valid attachment. Any dense progress reward or explicit failure
cost must be specified once and shared by every baseline. Keep calibrated task
outcomes distinct from the old uncalibrated force-based teacher score.

## Focused research extension: physically calibrated edit advantages

**October 3 status:** retain this relative-value loss as an auxiliary baseline.
The metric candidate below is also downgraded after the GPS/Q-loss literature
audit. Neither is a verified novel method or an established explanation of
current failures. Finish the bounded M4 baseline before committing to either
extension or a larger EXPO implementation campaign.

Research question: **Can a limited number of physical comparisons teach a critic
which local policy edits actually help, so online post-training improves more
reliably per simulator hour on new garments and moving humans?**

The hypothesized failure is inaccurate local action ranking under distribution
shift, not a proven universal failure of Q-learning. The editor can otherwise
exploit ranking errors and feed bad actions back into the base. Ordinary RL
must first establish whether this is material in the current task.

At a sampled policy state, fork the base action and one edited action under
normal physics. Use the same garment, complete initial state, human-motion
realization and absolute time. For a one-step critic, execute the differing
first actions, then use the same continuation-policy version for the remaining
H-1 decisions. Do not hold each first action for H steps unless defining a
different macro-action critic. Human future samples are simulation disturbances,
not inputs to the actor/editor/critic; evaluate held-out subjects and sequences.

Let $h$ be observable history, $a_0$ the base action, $a_1$ its edit, and
$\bar V$ a target continuation value under a fixed policy version. A paired
H-step target is

$$
\widehat\Delta_H =
\sum_{k=0}^{H-1}\gamma^k(r_k^1-r_k^0)
+\gamma^H[b_H^1\bar V(h_H^1)-b_H^0\bar V(h_H^0)].
$$

$b_H^j$ is zero after true termination and one for a nonterminal continuation;
absorbing padding is used for branches that terminate before H. Time-limit
semantics must match the finite-horizon task rather than default to bootstrapping.
The tail is estimated. Short physical rollouts do not establish full task value.
Forecast-model branches and actual-environment branches must be distinguished.

Add the supervised relative-value term

$$
\mathcal L_Q = \mathcal L_{TD}
+\lambda_b\,\mathbb E\left[
w\,\ell\left(Q(h,a_1)-Q(h,a_0)-\operatorname{sg}(\widehat\Delta_H)\right)
\right].
$$

Use a robust loss and empirical reliability weights. Measure uncertainty with
repeated continuations; use fresh validation after proposal selection. Shared
human motion does not make the IPC solver deterministic or guarantee variance
reduction. Require normal replay to anchor absolute Q; differences alone leave
state-dependent offsets undetermined.

Keep the EXPO editor and base updates unchanged in the first comparison. This
isolates the proposed critic supervision. Limit physical queries to a declared
budget. Start with fixed random queried states; uncertainty-triggered allocation
is a later ablation, not an extra untested component in the first method.

The formula is a difference of multi-step TD targets, **not new mathematics**.
Short model rollouts improving value estimates have direct precedent in
[Model-Based Value Expansion](https://arxiv.org/abs/1803.00101), and pairing
stochastic rollouts also has prior art, including
[Luck Is Not Skill](https://arxiv.org/abs/2609.24144). The potential contribution
must be a demonstrably better policy-improvement procedure under expensive,
noisy contact simulation, with enough specificity to outperform those controls.
Calling EXPO plus IPC a new algorithm is insufficient. No priority claim is made.

## New algorithm candidate: a shared consequence metric for editing and absorption

**Historical proposal, downgraded on October 3:** GPS already uses
teacher-precision-weighted imitation, and MPC Q-loss directly optimizes the
consequences of student actions. The following construction remains documented
for comparison, not selected as a novel contribution or the immediate experiment.

### Proposed contribution relative to EXPO

The proposed unit of research is the **policy-improvement update**, not another
physics mechanism study. Learn a local metric of how action errors change task
geometry, then use its inverse to shape residual exploration and the metric
itself to weight supervised absorption into the base. This couples where the
learner tries edits with where the base must reproduce improvements precisely.

EXPO already learns state-dependent Gaussian edits, including their means and
scales. It must not be described as fixed isotropic random exploration. Its
bounded action edits, value selection and supervised base updates are the
reference components. The proposed difference is direct consequence supervision
of a shared geometry used by both editor and base updates, rather than learning
all of that allocation implicitly from scalar returns.

The falsifiable mechanism is **anisotropic physical sensitivity**: equally sized
action changes may have very different effects on the cloth, and equal action
regression errors may have different execution consequences. This is plausible
in dressing, but has not been established for our current policy, and does not
establish that ordinary EXPO cannot learn the same allocation efficiently.

### 1. Learn the consequence metric from ordinary-physics branches

Let h contain causal point-cloud history and proprioception. Normalize action
coordinates using fixed physical action scales; do not mix raw meters and
radians in a Euclidean norm. Let z contain normalized task geometry, initially
cuff-to-arm transverse displacement, sleeve advancement, attachment tracking
and local deformation summaries. Training labels may use simulator geometry;
the deployed metric predictor receives only h. No force estimate or future GRAB
motion is an actor or metric input.

From one complete simulator state, compare small action perturbations under
the same human-motion realization. Start with one-decision responses. If H>1
is subsequently used, perturb only the first action and use the same fixed
continuation-policy version afterward. Current H=4 held-action candidate scores
are not observations from this one-action response experiment.

Fit a local response B, or its history-conditioned predictor B_psi:

$$
z_i-z_j \approx B_\psi(h)(a_i-a_j), \qquad
G(h)=B_\psi(h)^\top W B_\psi(h)+\lambda I.
$$

W fixes feature units and relative measurement scales in advance. Repeat a
subset of identical branches to estimate replay noise; invalid-physics branches
are excluded from physical response targets. Shared human motion does not make
IPC deterministic. Check the rank of the perturbation design: correlated
candidate actions cannot identify all six action directions. Current logs have
progress and attachment tracking, but lack the full geometry vector above.
They support an initial check, not a ready-made full metric dataset.

To isolate direction from overall edit size, regularize eigenvalues and define

$$
M(h)=\frac{G(h)}{\det(G(h))^{1/d}},
$$

where d is action dimension. Det(M)=1 gives equal-volume ellipsoids before
physical action limits. A scalar radius remains a separately tuned parameter.
Freeze and stop-gradient M within each learner update; refit as the policy
distribution changes. The first implementation uses M(h), independently of
the selected imitation target, rather than target-dependent loss weights.

### 2. Change the editor's optimization geometry

Keep EXPO's task reward, TD update formula and value-based selection rule. Use
the same modified candidate family for behavior and TD maximization. Replace
the reference edit geometry with

$$
\max_{\pi_e}\;\mathbb E[Q(h,a_0+\delta)]+\alpha\mathcal H(\pi_e),
\qquad \delta^\top M(h)\delta\leq\epsilon^2,
\qquad a_0+\delta\in\mathcal A.
$$

One parameterization is delta = epsilon M^(-1/2)u for u in the unit ball,
with the environment's action limits additionally enforced. An implementation
must account for transformations/truncation in the action density and entropy;
silently clipping a Gaussian is not an equivalent objective. Original base
actions remain candidates. The editor is still learned against Q.

For a linearized Q with gradient g, the unconstrained-by-actuator local solution
is epsilon M^(-1)g / sqrt(g^T M^(-1)g). This is a standard ellipsoidal
optimization identity, not a new theorem. It illustrates that useful directions
can receive larger changes when their predicted geometric effect is smaller.

High sensitivity is not synonymous with harm: an essential recovery may require
a large change in precisely such a direction. Measure whether the proposed
region excludes successful recoveries. A symmetric local metric cannot encode
arbitrary one-sided contact failures, disconnected feasible actions or changing
contact modes. It is an inductive bias for data efficiency, not a safety
certificate or a solution to every dressing failure.

### 3. Use the same metric when the base absorbs improvements

For a deterministic base and an improved action a*, use

$$
\mathcal L_{\rm absorb}
=\mathbb E[(\pi_\theta(h)-a^*)^\top
\operatorname{sg}(M(h))(\pi_\theta(h)-a^*)].
$$

For a conditional flow base, a compatible supervised surrogate is

$$
x_t=(1-t)\xi+t a^*,\quad e=v_\theta(x_t,t,h)-(a^*-\xi),\qquad
\mathcal L_{\rm FM,M}=\mathbb E[e^\top\operatorname{sg}(M(h))e].
$$

Keep the reference method's target selection and replay mixture the same. M is
positive definite, fixed given h, and independent of a*, xi and t within an
update. Under those assumptions, the population conditional regression optimum
is unchanged; weighting reallocates finite-capacity/finite-update fitting
effort. It does not guarantee a better generative distribution or exact
preservation of the physical consequences of the final sampled action. Test
the generated actions in the environment.

The design has one shared principle: **explore less in physically sensitive
directions, and fit accepted improvements more precisely in those directions**.
Report both base-plus-editor and base-only task success to distinguish better
online action selection from successful absorption.

At a fixed history this is interpretable as a linear change of action coordinates
by M^(1/2). The inverse/direct pairing is consequently not a new optimization
identity. The research burden lies in learning useful physical coordinates from
limited branch data, maintaining them through post-training, and showing a gain
over learned covariance and weighted-regression controls at equal total cost.

### Closest prior art and remaining novelty burden

| Prior | Existing contribution / overlap | What the proposed experiment must add |
|---|---|---|
| [EXPO](https://arxiv.org/html/2507.07986v3), [EXPO-FT](https://arxiv.org/html/2605.25477v2) | Learned edits, Q selection, stable supervised base updates | A consequence-supervised shared metric must improve the update at equal total cost |
| [Guided Policy Search, JMLR 2016](https://www.jmlr.org/papers/volume17/15-522/15-522.pdf), section 4.3 | Teacher covariance controls supervised precision; cost-sensitive directions receive more fitting weight | Inverse exploration covariance/direct imitation precision is not by itself a new coupling |
| [MPC exact Q-loss and Gauss-Newton approximation, CDC 2023](https://publications.syscop.de/Ghezzi2023b.pdf) | Fit actions through their cost-to-go, with a local second-order surrogate | Demonstrate a difference beyond consequence-weighted imitation and local quadratic approximations |
| [Task Space Exploration in Robot RL](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/PuzeLiu/MasterThesisJohannesHeeg.pdf) | Jacobian-based action sampling and covariance design already exist | Do not claim Jacobian preconditioning or anisotropic exploration as new; test the coupled exploration/absorption rule for pretrained policies |
| [Optimal Transport TRPO manuscript](https://openreview.net/references/pdf?id=YFOHPrGY1) | General transport costs define policy trust regions | A physical action metric is not a new general trust-region principle or theorem |
| [Online Safety Filter for Deformable Object Manipulation](https://arxiv.org/abs/2605.01069) | Learned consequence dynamics and a barrier filter; reported experiments concern fluids | Demonstrate learning/absorption gains over an explicit filter with the same consequence data; make no safety guarantee |
| [TaSIL](https://openreview.net/references/pdf?id=R2AxxFf9PR) | Derivative information improves imitation robustness | Distinguish environment action-to-outcome sensitivity from matching a teacher policy's state derivatives |

The candidate's narrow contribution would be a practical, consequence-supervised
post-training update that jointly allocates exploration and absorption precision
and demonstrably transfers across contact configurations. Generic metric
learning, preconditioning, weighted regression, imitation and IPC are existing
ingredients. This search does not establish priority for their proposed coupling.
If the result reduces to a tuned covariance or a safety filter with no additional
learning benefit, narrow or abandon the algorithm claim.

### Three next experiments and stopping criteria

These are retained candidate-specific tests, conditional on surviving the
October 3 prior-art and mechanism review above. They are not the immediate queue.

1. **Identify the mechanism before training a new policy.** Use existing logs
   for a cheap response/rank check. Then, if needed, propose a bounded set of 24
   recoverable saved states from multiple garment/body/motion cells, with at
   least 12 balanced perturbations plus an original-action branch per state.
   Repeat a subset to quantify simulator noise. Fit on some directions and
   evaluate fresh directions and magnitudes. Compare predicted outcome distance
   from action norm, a scalar sensitivity, diagonal G and full G. The state is
   the analysis unit; candidate pairs are correlated. Reject the mechanism if
   a reliable transferable directional effect is absent. This is a proposal,
   not a newly launched experiment or a promised runtime.
2. **Isolate improvement and absorption.** On fixed shared branch/replay data,
   compare a 2x2 design: ordinary/metric editor and ordinary/metric absorption.
   Match network size, update counts, candidate counts and observations. Include
   a scalar-radius editor, a learned full-covariance editor with the same hard
   action limits, and eigenvector-shuffled metrics at equal eigenvalues. Include
   an isotropic control using the same unit-ball parameterization as the metric
   editor, so a different squashing function does not explain a gain. Then use fresh rollouts;
   offline loss reductions alone do not establish policy improvement. If only
   a scalar radius helps, this proposal has not earned its directional claim.
3. **Run bounded online post-training.** Compare EXPO, EXPO with all the same
   extra physical transitions, a consequence-based safety-filter control, and
   the coupled method. Count every physical branch and all learning wall time.
   Use at least three training seeds initially; hold out garments and complete
   GRAB subjects/sequences, with unseen garments, unseen motion and joint shift
   reported separately. Bootstrap/paired comparisons at independent task-cell
   level, accounting for training-seed variability. Increase the evaluation
   set based on the minimum effect of interest, not significance chasing.

Primary endpoints are sustained dressing completion and success per total
simulator/learner hour. Secondary endpoints are attachment failure, harmful
edits, useful-edit discovery per physical query, and the base-only absorption
gap. Existing simulation attachment failure is a held-vertex tracking criterion,
not a measured real gripper slip probability. Claims of a general contact-rich
post-training method would also need a second task beyond dressing.

Moving arms and ClothesNet remain the two task axes. The proposed metric changes
with observed motion and garment geometry; its generalization is something to
measure, not assume. The current teacher/DAgger queue supplies a baseline and
potential initialization. No job was stopped, no EXPO learner was launched, and
no new success rate was obtained while constructing this proposal.

## Controls for the auxiliary relative-value extension

Use the same base, history, task splits, objective, edit family and total
simulation/learning budget:

| Method | Purpose |
|---|---|
| Existing dynamic teacher DAgger | Does reward-based policy improvement beat direct corrective imitation? |
| EXPO-style post-training | Does the established recipe already solve the problem? |
| EXPO plus physically calibrated edit advantages | Does the proposed supervision add value? |

Also give EXPO the same extra branch transitions without the relative-value
loss, and compare equal-cost unpaired extra transitions. These controls separate
the proposed supervision from simply purchasing more informative data. A
model-based value-expansion control shares the same H and continuation budget.

First measure held-out action-ranking error, harmful-edit rate, and calibration
against fresh physical continuation outcomes. Then measure full success and
success per simulator/total wall-clock hour. Report three training seeds and
paired task cells; task diversity is not replaced by many repeats of two clips.
Evaluate unseen garments, unseen motions, and both together.

If ordinary EXPO ranks edits correctly and improves reliably, use that result
and reject the supposed critic bottleneck. If the extension only beats DAgger
but matches ordinary EXPO, it is not an algorithm contribution. If the relative
loss is no better than identical extra transitions, it has not earned a novelty
claim. These outcomes refine the post-training method, not the application scope.

## Immediate work and status

Finish the existing bounded M4 student pilot and use the decision protocol at
the top of this document. M3 is terminal. M4 passed its teacher feasibility gate
and is collecting initialization episodes; no dynamic student result exists yet.
No new simulation or training was launched during this literature review.

If later evidence justifies an EXPO comparison, its first implementation unit
is still a shared replay/outcome interface with complete successor states,
rewards, termination and consistent temporal semantics, followed by the editor
and critic. The current candidate logs are not Bellman replay. The metric and
relative-value extensions remain unimplemented comparison ideas; the new
literature findings do not justify presenting either as a discovered algorithm.
