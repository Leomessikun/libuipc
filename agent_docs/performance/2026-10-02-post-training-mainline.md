# Dressing post-training: evidence, prior art, and the continuation decision

## Latest method-design response — 2026-10-03

The owner requests a creative, implementable post-training method. See the
[new interaction-goal transfer design](2026-10-03-garment-outcome-posttraining.md)
for the candidate algorithm, primary-source comparisons, data audit and
bounded development protocol. It proposes transferring local garment–body
interaction goals and re-solving their robot realization in the target
garment's physics. It does not claim that generic goal conditioning, repair
search or distillation is new. The specific transport operator's novelty and
advantage are research hypotheses, not results. No new GPU job was launched;
the stop instruction below remains in force.

## Current decision: withdraw implementation priority and stop the old queue — 2026-10-03

The owner challenged the value of the interaction-supervised proposal and
authorized stopping launched jobs when continuation is not justified. The B+C
proposal below remains a recorded hypothesis, **not the selected implementation
agenda**. Its decomposition reconstructs the ordinary dynamic action gain;
there is no established transfer advantage over a model given the same data,
and the additional physical queries have not earned their cost. Neither the
primary bottleneck nor a useful structural assumption has been established.

The M4 teacher/student pipeline was stopped under the owner's conditional
stop authorization. Its remaining budget would answer a conventional
distillation question, not establish the missing post-training contribution.
Preserve its results and partial logs; do not restart its queued collection,
training or 72-episode evaluation. Two complete successful initialization
trajectories were retained; no student was trained. See the operational record in
[the M4 report](2026-10-02-observed-motion-student.md).

The research target remains autonomous improvement on new garments and human
motions when the current policy and short-horizon labeler cannot supply useful
corrections. This is a capability target, not a new algorithm. Recovery learning,
MPC and RL already have substantial prior art. A replacement proposal needs a
specific justified learning mechanism and a credible total-cost advantage;
another renamed decomposition is not an adequate deliverable.

## Selected research proposal: learn how motion changes action usefulness — 2026-10-03

**Status: unimplemented candidate, no longer selected; see the decision above.**
This section originally superseded
the framing-only next step below. The selected question is whether explicitly
learning the interaction between a robot correction and human motion makes
post-training more data efficient and transferable across garments. It does not
restore the rejected optimizer-through-training proposal. No new simulator,
training or evaluation jobs were launched for this review.

The owner also requested deletion of obsolete training folders. That work is
complete: 47 retired experiment directories removed, 63.784 GiB reclaimed net,
with compact evidence retained and verified. See the
[cleanup manifest](2026-10-03-retired-training-cleanup.json). Current dressing
data, baselines and M4 dependencies remain available.

### 1. The proposed contribution, in operational terms

> Post-train an existing dressing policy by learning how human motion changes
> the relative usefulness of its possible corrections. Obtain this supervision
> through crossed robot-action and human-motion interventions from identical
> physical states, reuse static interaction data, and marginalize uncertain
> human futures before producing deployable policy targets.

For example, a small forward pull can help when a sleeve can slide, but become
counterproductive when arm motion tightens the fabric. The learner needs to
change which correction it prefers in response to that interaction. A predicted
arm position alone does not specify the cloth response. This is a physical
hypothesis, not a diagnosis established by our grasp-tracking failures.

The intended technical deliverable is an **interaction-supervised policy
improvement operator** with a specified data construction, model and update.
The scientific claim would be that it needs fewer dynamic physical queries than
an otherwise matched unstructured outcome model or winner-action imitation.
Transfer to unseen garments is a separate required result. The four-term
subtraction, residual networks, history conditioning and weighted imitation
are established tools; none is an original mathematical claim here.

### 2. What the existing evidence actually motivates

| Evidence or constraint | Consequence for this proposal |
|---|---|
| Static r1 improved over its initialization; substantially more static labels did not consistently help r2 | Preserve the useful initialization. This does not identify the cause of saturation |
| Corrected M4 validation has four valid causal-teacher successes over two clips; current-pose planning has two | Dynamic information can matter in this small pilot; it does not establish a general contact mechanism |
| One current planning query evaluates 12 candidates and takes minutes | Count every branch, reset, render and continuation; data construction must justify its extra cost |
| Old static datasets include observations, cloth vertices, actions and geometric task signals | Reuse them for initialization, geometry supervision and compatible static outcome targets |
| The current dynamic trajectory format saves observations/actions, while its planner log saves candidate outcomes under one motion model | These files do not supply matched crossed interventions under different motion continuations |
| The deployed input is partial visual history | Future ground truth may supervise training, but cannot be passed to the actor or used to choose a different immediate action in each hypothetical future |

Full FMVP already uses real visual/force post-training for moving arms. Our
comparison checkpoint `fmvp_sim` is only its simulation component. An eventual
real-data-efficiency claim must compare against the full method's training
setting. [FMVP, sections 3–5](https://arxiv.org/html/2509.12741v1).

### 3. Why separate static and motion data can be insufficient

Consider the illustrative scalar outcome family

$$
Y_c(u,m)=u+\tfrac12m+c\,u m,
$$

where `u` is a correction relative to the base action and `m` denotes a change
in human continuation. Every value of `c` gives identical data on both axes:
`Y_c(u,0)=u` and `Y_c(0,m)=m/2`. Nevertheless, the benefit of `u=1` at `m=1`
is `1+c`: positive for `c=0` and negative for `c=-2`.

Thus unlimited data on those two axes alone cannot identify the interaction
without a structural assumption. Joint interventions supply the missing
information. This elementary example is an identifiability illustration,
**not a new theorem, a measured dressing result, or proof that existing dynamic
manipulation methods fail**. Ordinary dynamic data can also identify the
interaction with sufficient coverage; the proposal concerns query efficiency.

### 4. State, interventions and the target to learn

Let `s` contain the simulator's cloth configuration/velocity, articulated body,
grasp state and controller state. Let `h` contain only available point-cloud,
robot-pose and past-action history. Let `a0=pi0(h)` be a frozen reference-policy
action, initially r1. Candidate actions `{a_j}` include `a0` and common bounded
corrections. Candidate construction is identical across motion branches.

Let `m` specify a short human joint trajectory and `m0` a reference continuation
from the **same current state**. For an initially stationary arm, `m0` can remain
stationary. For an already moving arm it must preserve the initial position and
velocity and decelerate smoothly; instantaneously freezing a moving body would
confound the target with a reset impulse. Log both trajectories and check body
tracking independently of task outcomes.

Represent `m` by the parameters of a continuation law relative to the current
body, with `m0` the fixed braking/reference law. Its physical realization uses
the simulator state only inside training branches. The policy does not receive
hidden joint coordinates or true velocities through a reference-motion token.
The learned motion distribution and branch sampling must cover the deployment
conditional distribution; arbitrary motion randomization need not produce
correct posterior action values under occlusion.

Define `Y_H(s,a,m)` as the expected bounded task outcome of executing `a` for
one decision, followed by the same frozen, causal continuation policy for
`H-1` decisions, under human continuation `m`. Start with `H=4`; record the
whole outcome trace. The initial scalar target uses normalized sleeve progress
and an explicitly versioned grasp-validity penalty, with geometric validity
reported separately. It does not require matching simulated forces. A solver
or body-tracking failure gives a missing/invalid sample, not a bad-action label.

This is a **short-horizon outcome**, not an optimal Q-function or a guarantee
of final dressing success. Backing off with benefits beyond `H` can still be
misranked. A later terminal-value extension would require additional, correctly
aligned supervision; it is not silently assumed in this proposal.

Collect the following four outcomes from an identical complete snapshot:

| | Reference continuation `m0` | Alternative continuation `m` |
|---|---|---|
| Base action `a0` | `Y00` | `Y01` |
| Candidate action `a` | `Y10` | `Y11` |

Define the reference action gain and motion-induced change in that gain:

$$
B(s,a)=Y_{10}-Y_{00},\qquad
C(s,a,m)=Y_{11}-Y_{01}-Y_{10}+Y_{00}.
$$

Then the dynamic gain is exactly

$$
A(s,a,m)=Y_H(s,a,m)-Y_H(s,a_0,m)=B(s,a)+C(s,a,m).
$$

`C` asks whether human motion changes the benefit of choosing this correction.
It is not a force estimate, a contact label or merely an extra input feature.
Nonzero `C` can also arise from geometry or a nonlinear score; it does not by
itself prove contact-mediated coupling. Include an articulated geometric
transport baseline to test whether simple motion compensation explains it.

The identities hold for expectations. Independent repeats are still needed to
estimate uncertainty; restoring a seed does not make IPC deterministic. Reuse
shared baseline branches, but retain covariance when computing contrast errors.

### 5. Model and learning losses

Use the same history encoder and parameter budget in the structured and
unstructured comparisons. Garment geometry and observed cloth motion belong
in `h`; garment IDs or hidden simulator contact states are not deployment
inputs. Arm-relative coordinates and material sleeve-section supervision are
reasonable shared preprocessing, not new contributions.

Parameterize observable reference gain and interaction as

$$
\widehat B_\phi(h,a)=b_\phi(h,a)-b_\phi(h,a_0),
$$
$$
\widehat C_\psi(h,a,m)=
g_\psi(h,a,m)-g_\psi(h,a,m_0)
-g_\psi(h,a_0,m)+g_\psi(h,a_0,m_0).
$$

This enforces zero gain for the reference action and zero interaction on either
reference axis. It is an anchored functional decomposition, not a novel identity.
No low-rank assumption is required. Low-rank payoff models already exist and
contact transitions need not be low rank; that extension is not selected.

With robust regression loss `ell`, train

$$
L_{\rm outcome}=
\sum_i w_i\{\ell(\widehat B_i-B_i)
+\lambda_C\ell(\widehat C_i-C_i)
+\lambda_A\ell(\widehat B_i+\widehat C_i-A_i)\}.
$$

`w_i` masks invalid physical branches and caps any precision weighting derived
from repeat variability. A near-zero contrast stays near zero; it must not be
normalized into a large advantage. The last term checks reconstruction of
actual dynamic action gain. A matched control fits `A` directly with the same
data, capacity and optimization budget. Removing the `C` term tests whether
motion-dependent interactions are useful at all.

With unlimited capacity and data, a full outcome model can reconstruct the
same contrasts. There is no claim of a more expressive hypothesis class. The
proposed benefit must come from finite-budget supervision and transfer. If the
loss merely behaves like an alternative regression weighting without improving
that tradeoff, it is too weak to carry the paper.

Partial observability means these models estimate conditional averages given
`h`, not the exact hidden cloth state. If important interactions remain visually
indistinguishable, more privileged labels cannot make a visual actor recover
information it does not possess.

### 6. How this updates the existing policy

Fit a causal motion distribution `p_omega(m|h)` using training-subject GRAB
sequences and observed motion histories. A finite-difference forecaster with
training-calibrated residual samples is the simple starting baseline. Any
learned forecaster must use the same observations in all policy comparisons.
Evaluate the consequence model at the same possible **immediate action** for
every sampled future, then average:

$$
\overline A_j(h)=\widehat B(h,a_j)+
\mathbb E_{m\sim p_\omega(\cdot|h)}\widehat C(h,a_j,m).
$$

Do not optimize a different immediate action for each true future and then
average those oracle actions. The base continuation also receives only causal
observations. This preserves the information available at deployment.

Choose a positive proposal prior `mu_j` over the finite candidates, biased
toward small edits around `a0`. It is a specified proposal distribution, not
an invented tractable density of the implicit FMVP/flow policy. Form targets

$$
q_j^*(h)=\frac{\mu_j\exp(\overline A_j/\tau)}
{\sum_k\mu_k\exp(\overline A_k/\tau)}.
$$

This is the standard solution of a KL-regularized improvement problem. The
proposed change is the physically supervised construction of `overline A`, not
the exponential weighting. Fit the policy to these weighted **action samples**
and retained successful static examples. Do not average incompatible corrections
into a single MSE target.

For the existing flow-policy family, a concrete update is positive-weighted
conditional flow matching:

$$
L_\pi=\mathbb E_{h,j\sim q^*,z,t}
\|v_\theta((1-t)z+t a_j,t,h)-(a_j-z)\|^2
+\lambda_{\rm keep}L_{\rm static},
\qquad \theta\leftarrow\theta-\eta\nabla_\theta L_\pi.
$$

Use `t~Uniform[0,1]`, Gaussian `z` and the common action normalization. A
categorical selector over corrections provides a simpler implementation control.
The flow backbone itself is not the contribution. At deployment only the
post-trained history policy runs; IPC and hypothetical human futures are
training resources. This replaces hard winner-action targets with an
interaction-supervised improvement distribution, without requiring a complete
successful privileged-teacher trajectory for every garment/motion pair.

The local KL update has no automatic global success guarantee. A simple useful
bound is: if every candidate's estimated expected gain differs from truth by at
most `epsilon`, the estimated greedy candidate loses at most `2 epsilon`
against the true best candidate. This familiar bound says why gain/ranking
accuracy matters; it is not a new theoretical result or a claim about final
episode return.

### 7. Algorithm and exact reuse of existing data

1. Preserve r1, the existing flow initialization and static success anchors.
   Warm geometry features and compatible reference outcome targets from existing
   data. Record horizon, controller, score version and continuation semantics.
2. Visit training garment/body states with the current policy. At selected
   states capture complete physics, controller and observation-history snapshots.
   Cross a common candidate-action set with a small set of plausible human
   continuations; share the baseline row/column and retain all branch outcomes.
3. Fit `B` and `C`, with held-out garments/sequences for model selection. Train
   the causal motion model on training subjects only.
4. Marginalize motion futures to obtain `q*`; update the initialized policy with
   weighted imitation plus static retention. Revisit states with the updated
   policy only within an explicitly budgeted next collection round.

Actual files inspected in this review:

- Static v5 sample `tshirt_4/.../body_3033_seed_2026292600/baseline_rep1.npz`:
  235 observations, 234 six-dimensional actions, 5,761 cloth vertices per frame,
  executed translations, validity and geometric sleeve signals. These can
  support representation/outcome supervision, subject to controller and score
  compatibility. They are not complete restorable IPC snapshots.
- Cloth3D DAgger `lookahead.jsonl` files retain all candidate actions and their
  progress/tracking/feasibility values, not just selected labels. These are
  useful static supervision when their horizon and scoring contract match.
- M4 `validate_pass_causal_rep0/causal_trajectory.npz` has 451 observations and
  450 actions/teacher actions, with query flags. Its planner log evaluates one
  motion belief at a time. Separate condition rollouts are not matched snapshot
  interventions and cannot be subtracted to fabricate `C`.

Current `MotionPlanner.evaluate` **holds the candidate for all H decisions**.
The proposed first-action-plus-causal-continuation target has different semantics.
It needs a separate query path and complete history restoration; old H=4 labels
cannot be relabeled as that target. Legacy H=1 data can warm compatible heads,
but are not H=4 supervision. Do not edit or reinterpret the live M4 pipeline.

For K candidates including the base and M human continuations including the
reference, a full block costs **K*M branches**, each of H simulator decisions,
plus setup, rendering and roll-in. A two-by-two contrast costs four outcomes;
sharing anchors avoids naive repeated evaluation, but does not make the data
free. Paired collection can cost more per state than ordinary labels. The
method must earn that cost by needing fewer states/episodes. No speedup or
sample-complexity theorem is established.

The new record contract must contain `snapshot_id`, snapshot/history hashes,
garment/body/motion split IDs, reference-policy hash, common candidate array,
continuation-law parameters, H and action duration, continuation-policy hash,
score version, outcome/validity traces, repeat ID, random state and wall time.
`snapshot_id` must identify the full actual restored state, not just a seed or
the first cloth positions. Unpaired legacy records remain separately labeled.

### 8. Prior-art attack and the remaining claim

| Primary source and scope read | Existing contribution | Difference this proposal must demonstrate |
|---|---|---|
| [EXPO, section 4](https://arxiv.org/html/2507.07986v3) | Gaussian edits, Q-based selection/TD backups, supervised absorption into an expressive policy | A motion-interaction target and acquisition design improve the quality/cost of corrections. Editing and absorbing actions are not new |
| [WISE, sections 3 and appendices](https://arxiv.org/html/2609.03681v1) | Scheduled bounded counterfactual imagination and policy refinement | Cross interventions on human continuation and robot correction identify reusable interaction supervision; merely replacing its world model with IPC is insufficient |
| [SIDO, sections 4.1–4.3](https://arxiv.org/html/2607.27890v1) | Static-data action morphing preserves hand-object relative pose; dynamics-aware variant models robot tracking | Learn changes in action usefulness during sustained deformable interaction; compare against its geometric idea with the same motion information |
| [GDM, sections III–IV](https://spiral.imperial.ac.uk/bitstreams/0da51527-b0f6-4c04-a2c5-0203c612982e/download) | Partial-cloud garment-opening dynamics and MPC, trained iteratively in simulation; assumes a stationary recipient | Learn policy correction gains under exogenous motion and amortize them. Predicting an opening or using MPC is not new |
| [IADD-TR, section III](https://arxiv.org/html/2608.10634v1) | Zero-action-anchored action/intermediate-state/natural-evolution factorization, targeted actor-critic regularization | Direct crossed outcome supervision for two interventions rather than identifying its latent two-stage dynamics. Anchoring causal models itself is not new |
| [Deep Coordination Graphs, sections 1–2](https://proceedings.mlr.press/v119/boehmer20a/boehmer20a.pdf) | Pairwise payoff/value factorization, parameter sharing and low-rank models in cooperative MARL | Human motion is exogenous and marginalized, with measured crossed branch targets. Pairwise interaction terms or low rank cannot be claimed as new |
| [Structure Detection for Contextual RL, section 3](https://ojs.aaai.org/index.php/AAAI/article/download/40137/44098) | Decomposes policy-transfer performance into source, target and interaction terms to select training tasks | Our quantity is within-state action gain and our output updates a deployable policy. ANOVA-style decomposition or structure-aware task selection is not original |
| [DPP, method and limitations](https://arxiv.org/html/2609.33172v1) | Counterfactual observation/context planning reuses static skills for moving targets | A strong simpler competitor; improvements must require learning additional interaction response, not just eliciting an existing skill |
| [Dressing in Motion](https://arxiv.org/html/2609.04759v1) and [FMVP](https://arxiv.org/html/2509.12741v1) | Reactive motion adaptation, or real force/vision post-training | Dynamic dressing itself is established. Require an explicit data-efficiency/generalization comparison |

The review found substantial overlap in the components and **did not establish
worldwide novelty**. The remaining defensible candidate is the complete
cross-intervention training procedure and its garment/motion transfer benefit.
If a matched unstructured learner achieves the same performance and cost, the
extra decomposition has not earned a method contribution. Changing notation or
calling the contrast causal does not rescue it.

### 9. Candidate hypotheses considered

| Observation → mechanism → method → prediction | Decision |
|---|---|
| Dynamic failure → missing history → privileged teacher/history distillation → history helps | Existing M4 baseline; substantial prior art, not the new contribution |
| Target motion → pose mismatch → geometric action/observation transport → static data suffice | Strong simpler baseline; SIDO/DPP already cover the broad idea |
| Occluded cloth → missing dynamics → predict complete opening and plan → better decisions | GDM already provides the broad method |
| Costly queries → low-rank response → matrix completion → fewer physical branches | Low-rank RL/DCG prior art and unverified contact-rank assumption; not selected |
| Motion changes correction utility → static effects fail to compose → crossed interaction supervision → better action ranking and post-training at equal cost | Selected, conditional on direct comparison with unstructured outcome learning |
| Privileged labels disagree under identical histories → unobservable interaction → belief/active sensing → informative actions help | Real possible limitation, but not diagnosed here; do not invent it to justify a new module |

### 10. One bounded new-method experiment, not another teacher rerun

This is an implementation protocol, **not a newly launched experiment**. The
first deliverable is the crossed-branch data contract and learner above. Do not
append further r1/current-pose/GICP repetition campaigns while designing it.

**Initial data budget:** 24 training/validation root states spanning eight
training-pool garments and three dressing stages, four candidates including
the base, three human continuations including the reference, and two physical
repeats: at most 576 branches, 2,304 H=4 decisions before roll-in/setup cost.
Keep two of those garments exclusively for model validation. Use training
motion subjects; final test garments, bodies and motion subjects are separate.
This is a small development set, not sufficient evidence of population success.
A repeat quantifies local disagreement; two repeats do not certify reliability.

**Three matched learners:** (i) same paired data, fit dynamic gains directly;
(ii) reference gains plus the learned interaction; (iii) remove the interaction
and retain the same history/motion information in the common geometric baseline.
Keep a winner-action imitation control on the same measured candidates. Use
three learner seeds, identical policy architecture and initialization. Include
EXPO as an algorithm baseline for a subsequent paper-scale study; do not call
these local supervised controls full EXPO reproductions.

**First measurements:** held-out candidate regret/rank reversals, error on
gain differences, invalid prediction rate, total simulator/CPU wall time, and
static retention. Split by garment/trajectory, never by neighboring frames.
Compare paired collection with an equal-total-cost ordinary collection control;
same-label-count alone is not a fair cost comparison. All H=4 continuation
render/inference cost counts.

**Final closed-loop comparison:** once this method is implemented, freeze one
test matrix crossing unseen garments with unseen body/motion sequences. Report
success and geometric progress, whole-episode attachment validity, and runtime.
Cluster uncertainty by garment and motion sequence; report learner seeds
separately. Test a prespecified practical improvement margin at matched cost,
with paired intervals. Do not treat adjacent states or two repeats as independent
tasks or insist that every pilot clip must succeed. A real-transfer claim still
needs robot evidence and a budget for real adaptation.

**Decisive failure conditions:**

1. Motion does not change useful action rankings beyond repeat variability after
   geometric compensation: the selected mechanism is not useful in this regime.
2. A direct outcome model matches the structured model on held-out ranking and
   closed-loop performance at equal cost: no benefit from the proposed operator.
3. Gains require future ground truth, hidden simulator inputs, favorable data
   filtering or different candidate actions: no deployable post-training claim.
4. Gains disappear on held-out garments: a motion result only, not the original
   garment-by-motion generalization contribution.
5. Extra branch/continuation cost exceeds the data savings: no efficiency claim.

The next three development actions are therefore: implement the complete
crossed-branch record/restore interface; implement the structured and direct
outcome learners plus identical policy updates; run one frozen-budget study of
the new operator. Existing M4 remains a bounded application baseline. No new
method, dynamic student result or publication-level improvement is claimed by
this research specification alone.

## First-principles correction — 2026-10-03

**The owner rejected the finite-update/joint-correction proposal below. It is
retired as the selected development direction.** Its assumed fitting-interference
bottleneck was not established in this task. Do not implement it or use a new
evaluation campaign to defend it. This revision changes the research framing
and documentation only; existing jobs and data are unchanged.

### What the task actually requires

Single-sleeve dressing requires moving a deformable garment into the correct
enclosing relationship with an articulated human arm through a feasible sequence
of contacts. The robot acts at a limited grasp region; the rest of the garment
responds indirectly. A geometrically correct gripper target does not uniquely
determine sleeve deformation, sliding, or whether a fold catches. Human motion
changes those relationships while also changing the target geometry.

Garment variation changes the same interaction: sleeve dimensions, compliance,
folds and the connection to the rest of the garment affect what can slide where.
Thus garment generalization and motion handling are coupled through garment-body
interaction. Treating them only as more task IDs or pose offsets misses this
structure. This is a task analysis, not a claim that registration or generic
policies can never solve particular cases.

Observations constrain what can be learned. If two hidden configurations give
the same available observation history but require different actions, a visual
history policy cannot reliably distinguish them without another informative
observation or interaction. A privileged teacher does not remove that limit.
Likewise, an unexpected motion cannot be predicted before any informative cue.
Robust response and pre-onset anticipation are different capabilities.

### Three corrections to the current evidence story

1. **The full FMVP method already addresses moving arms.** Its real-world
   post-training uses both vision and force. `fmvp_sim` is its simulation-trained
   component; showing that this checkpoint fails does not establish that the
   complete prior method cannot handle dynamic dressing. The potential gap is
   how much real dynamic interaction data are needed and what a useful dynamic
   simulator can transfer. See [FMVP, sections 3-5](https://arxiv.org/html/2509.12741v1).
2. **Current grasp loss is an attachment-tracking proxy.** The environment used
   by the motion probe sets `grasp_valid` by comparing maximum held-vertex target
   error against 0.02 m. See
   `/home/ge47gax/kun/libuipc-anticipatory-dressing/python/uipc_manip/dressing_env.py`
   (`grasp_tracking_tolerance_m`, `_tracking_error`, and the `grasp_valid`
   assignment). This is useful simulation bookkeeping, but not a measured real
   gripper slip, a diagnosed sleeve snag, or a human-comfort threshold. The
   separate human-body tracking tolerance is 0.002 m. Do not conflate them.
3. **Unreliable simulated force does not make real force uninformative.** FMVP
   explicitly uses real force to help with occluded interactions. A geometry-only
   policy is a sensing assumption to justify, not a deduction from noisy force
   readout in IPC. No sensor or runtime interface is changed by this review.

### Recommended contribution target

The strongest project-specific target is:

> Use simulated dynamic garment-body interaction to post-train a dressing
> policy that transfers its interaction handling to unseen garments and human
> motions, with less real dynamic dressing data, including appropriate recovery
> when following the arm alone is insufficient.

This states a **desired contribution**, not an achieved result or a verified
unoccupied literature gap. A useful demonstration would involve physically
different situations that require different choices: advancing when cloth can
slide; changing direction or temporarily backing off when continued pulling
worsens the configuration; and resuming dressing after the relationship becomes
favorable. Recovery, prediction, or a pause primitive alone is not novel.

The key method question becomes: **what action-relevant interaction information
can be learned in simulation and inferred at deployment, despite garment
variation, occlusion and inaccurate simulated forces?** One task-grounded
hypothesis is to supervise an observation/action-conditioned representation of
relative garment-body progress and sliding/obstruction transitions. Full mesh
state can supply training supervision; the deployed policy must infer only what
its sensor history supports and retain uncertainty where it cannot. ClothesNet
then varies the geometry, while GRAB varies the exogenous motion.

This is a possible learning target, not a selected architecture or a claim of a
new loss. Exact force reconstruction, full cloth reconstruction and privileged
action imitation are different possible targets with different transfer costs.
The scientific burden is to show why the selected representation improves
physical task decisions and transfer compared with direct policy post-training
and established predictive controllers. Merely adding contact labels, a graph,
or a future-prediction head does not establish that contribution.

### Relevant task prior art and the remaining burden

| Primary work | What it already establishes | Implication |
|---|---|---|
| [One Policy to Dress Them All, RSS 2023](https://roboticsproceedings.org/rss19/p008.pdf) | Learned dressing over garment and pose variation | Garment diversity itself is not new |
| [FMVP](https://arxiv.org/html/2509.12741v1) | Static simulation pretraining followed by real visual/force post-training for arm motion | Compare the complete method and real-data cost, not only `fmvp_sim` |
| [Dressing in Motion](https://arxiv.org/html/2609.04759v1), section V-B | Static-demonstration policy plus motion adaptation; simulation and real policies trained separately | Transferring dynamic simulation learning is a different claim from reactive registration, but must actually be shown |
| [Deep Haptic MPC, ICRA 2018](https://sites.gatech.edu/hrl/haptic-mpc/) | Predicts garment forces on the body and mitigates catches around fists/elbows | Contact prediction and avoiding snags are established |
| [Garment Diffusion Models, RA-L 2025](https://spiral.imperial.ac.uk/entities/publication/a0176f43-5eb7-4530-93d8-35c0118514df) | Partial-observation opening dynamics, MPC, and transfer to new garments/body configurations | Predicting garment dynamics under occlusion alone is not a new method; this entry uses the official abstract |
| [Active Boundary Component Models, IROS 2016](https://noah.nrw/ubbihs/download/pdf/5133742) | Tracks garment openings with constrained boundary models | Focusing on sleeve openings or garment topology alone is established |
| [Sparse Meets Dense, 2026 preprint](https://arxiv.org/html/2608.01083v1) | Task/contact-aware correspondences and constraint control for rigid-deformable manipulation | Relational garment-object representations are not new by themselves. Its reported tasks include hanger/bag insertion rather than dynamic human dressing |

A publication-level result could combine a new, justified transferable
interaction representation/learning mechanism with convincing dynamic dressing
and real-data-efficiency evidence. It need not invent a universal RL optimizer.
Conversely, IPC plus GRAB plus existing distillation is presently an infrastructure
and application pipeline, not an established method contribution.

### What the existing project can and cannot support

Existing rollouts, checkpoints and dynamic simulation are useful starting
assets. They do not yet show generalized contact recovery, real comfort, or
reduced real-data requirements. The fixed grasp/action interface also cannot
support claims of learned regrasping without additional functionality. Current
results concern one sleeve; they do not establish complete torso/bilateral
dressing. GRAB sequences prescribe arm motion and do not model how a person
changes motion in response to the robot.

The next intellectual deliverable is a task-grounded interaction representation
and learning target, with a clear difference from the predictive/correspondence
methods above. Do not start another teacher-versus-r1 table to manufacture this
difference. Do not assume the snag/sliding hypothesis is diagnosed by the grasp
proxy. This first-principles correction authorizes no new simulation campaign
and does not restore the rejected contact-friction mechanism study.

## Method development takes priority — 2026-10-03

**Retired candidate:** the owner subsequently rejected the method proposed in
this section. Preserve it as research history, not the next implementation plan.

**Latest owner instruction:** create a concrete post-training contribution and
investigate its prior art now. Do not keep expanding evaluations of existing
controllers. The earlier sequence of finishing M4, adding simple controls, then
running a mechanism diagnostic is superseded as the immediate research agenda.
The already authorized bounded M4 pipeline remains a baseline; it is not a
prerequisite for the method work below. This revision launches no simulations,
training, evaluations, or additional monitoring processes.

**Assessment:** the implemented planner/history-student/DAgger combination is
not yet a defensible new algorithm. The following is a concrete method candidate,
not a claim that its novelty or effectiveness is established. Its technical
focus is **joint selection of corrections using their effect on the updated
policy across states**, including states from other garments and motions.

### Research question and evidence boundary

Can a small physical-query budget improve a pretrained visual controller more
effectively when corrections are selected through the learner's actual update,
rather than selecting each teacher action independently and subsequently fitting
them all? In particular, can this retain useful motion corrections without
damaging previously successful garment configurations?

The existing evidence motivates expensive-interaction post-training: some
physical corrections help, label scaling has not consistently improved the
policy, and online planning is expensive. It **does not establish** that fitting
interference causes r2 saturation or that dynamic M4 students fail to absorb
their teacher. A weak teacher, limited action coverage, frozen features, short
planning horizons and ordinary optimization are competing explanations. The
candidate addresses fitting interference only when physically useful alternatives
and sufficient observable information exist. It cannot repair an unreachable
start or manufacture useful supervision from indistinguishable branch outcomes.

### Concrete proposal: choose corrections jointly through the learner

Keep the current pretrained controller and a trainable adapter initially. Let
`h_i` be a deployable observation history and `s_i` a saved simulator state used
only for training. Keep several candidate corrections at each training history,
including the original action. A slightly lower-scoring correction can be
preferable if the resulting shared policy executes it accurately and preserves
other useful behavior.

The procedure has two feedback paths:

1. **Physical response:** what happens if a particular action is executed at
   a saved state, followed by a specified continuation?
2. **Learning response:** after training on a proposed correction, how do the
   policy's actions change at *all* sampled histories, including other garments?

Use the learning response to propose a joint correction batch; use actual
closed-loop physical branches of the updated visual policy to refine that
proposal. The physical query evaluates what the student will execute. There is
no requirement to build a stronger privileged teacher first.

#### Finite update and the proposed search variables

Let `Z = {z_j}` denote correction targets at fixed, real training histories.
Let `U_K(theta, optimizer_state; Z, D_anchor)` be exactly K updates of the chosen
supervised post-training routine, including its anchor mixture and optimizer
state. Define:

$$
\theta_Z=U_K(\theta,m;Z,D_{\rm anchor}),\qquad
A_i(Z)=\pi_{\theta_Z}(h_i).
$$

`Z` is a set of optimization targets; it need not equal an expert's favorite
action. Every executed action still passes through the actual controller and
its actuator limits. Start with a small number of target directions derived
from the existing correction candidates, not unconstrained edits to a large
visual network. Retain an explicit no-update alternative: setting targets to
current actions is not necessarily a no-op with Adam momentum or weight decay.

For one plain SGD step on unnormalized squared error, without anchors, the
first-order learning response is

$$
A_i(Z)\approx\pi_\theta(h_i)
 -\eta\sum_j J_iJ_j^\top(\pi_\theta(h_j)-z_j),
\qquad J_i=\frac{\partial\pi_\theta(h_i)}{\partial\theta}.
$$

This is an existing neural tangent kernel identity, not a new theorem. Its
off-diagonal blocks explicitly describe how a correction at history j changes
the action at history i. With several Adam steps, use the derivative of the
actual finite training computation or exact cloned updates; do not substitute
the SGD identity and call it exact. Matrix-vector products avoid materializing
the full stacked Jacobian or every pairwise response block.

#### A concrete proposal rule and the physical outer objective

Use local action-consequence fits `q_hat_i` with a fixed base-policy continuation
to propose targets jointly:

$$
Z_{\rm prop}=\arg\max_{Z\in\mathcal Z}
 \sum_i w_i\widehat q_i(A_i(Z);\pi_\theta)
 -\lambda\sum_{i\in\mathcal I_{\rm anchor}}
 \|A_i(Z)-\pi_\theta(h_i)\|^2.
$$

The fitted `q_hat_i` are finite-horizon local outcome surrogates, not intrinsic
action quality and not a claim of critic-free learning. Their continuation,
human-motion realization, time step and horizon must be recorded. Restrict
optimization to supported regions; unqueried actions require new physical
feedback rather than confident interpolation across contact failures. An action
distance penalty at anchor histories is a trust mechanism, not a guarantee of
retained task success.

To correct the surrogate, score the resulting *whole policy update* with
training-state branches:

$$
F_H(Z)=\mathbb E_{(s,h)\sim\mu_{\rm train},\,\omega}
 [G_H(s,h,\pi_{\theta_Z};\omega)
  -G_H(s,h,\pi_\theta;\omega)]-\lambda C_{\rm anchor}(\theta_Z,\theta).
$$

Here `G_H` is the discounted task return of H **closed-loop** decisions, including
grasp termination. `omega` includes future human motion and simulator randomness.
The visual policy receives fresh observations and a correctly restored history
on each branch. GRAB continuation is environment input during training; it is
not provided to the student. Matching an external motion schedule does not make
CUDA/contact randomness identical. A short-horizon score is only a surrogate
for complete dressing, even if all branches are physically valid.

The proposed practical solver is to alternate cheap joint target proposals with
bounded physical feedback on their actual finite updates. Crucially, a
fixed-base-continuation action score and an updated-policy closed-loop score are
different quantities. Do not pool the latter into `q_hat_i(action)` as though
the continuation were unchanged. Instead retain an update-level discrepancy:

$$
f_{\rm local}(Z)=\sum_i w_i[
 \widehat q_i(A_i(Z);\pi_\theta)
 -\widehat q_i(\pi_\theta(h_i);\pi_\theta)]-\lambda C_{\rm anchor},
\qquad e(Z)=F_H(Z)-f_{\rm local}(Z).
$$

Fit a local residual `e_hat(Z)` only in the small target-coordinate neighborhood
already queried, then propose with `f_local + e_hat` inside that neighborhood.
With too few points for such a fit, use the discrepancy to shrink the proposal
region rather than invent a confident correction. This residual accounts for
both the action-surrogate error and changed continuation; it is not an unbiased
estimator or a generalization guarantee. Keeping the previous policy as a
candidate, commit the selected cloned policy **and its optimizer state**.
Do not average the weights of several tested policies into an untested policy.
This is a local surrogate optimization procedure, not a global optimality or
monotonic-improvement guarantee.

```text
snapshot the learner, optimizer, and a small batch of training states/histories
retain alternative corrections and old-task anchors
compute the finite learner response to correction targets
jointly propose a target batch using local physical outcome fits
clone the real learner update and execute its visual policy in training branches
feed observed outcomes back into the proposal; obey a fixed interaction budget
commit one selected update, or the unchanged learner
```

The cross-state coupling and the physical feedback must both be present in the
candidate implementation. Merely selecting a successful teacher action, adding
data weights, or performing a line search over the BC learning rate does not
implement the proposed solver.

This solver inherits ordinary model-management/trust-region ideas; the
discrepancy correction itself is not a novelty claim. Its practical question is
whether coupling corrections before purchasing expensive policy branches makes
the search more useful than independent target selection or direct guided ES.

#### Fit to this repository and computation budget

The current `train_dynamic_student.py` trains on CPU with Adam, gradient clipping,
per-episode/query weights and a squared-logit-residual penalty. Its world action
includes tanh, coordinate rotation and clipping. A functional update must match
all of those operations. `dynamic_student.py` already supplies the four-frame
adapter and frozen encoder interface. Student checkpoints currently contain the
adapter weights and configuration, not a resumable Adam state; a new ongoing
post-training runner must either explicitly initialize Adam or save its state.

`motion_lookahead_probe.py` currently holds a candidate action for H decisions
and skips branch observations. A branch executing an updated policy must build
observations and preserve each branch's observation history. Existing snapshots
restore simulator/controller/motion state; they do not automatically restore an
external student's deque. A first implementation can use the exact small adapter
update and cached training features; large-encoder or flow-policy hypergradients
are not required to investigate the proposed operator.

For B proposed updates, S training snapshots and horizon H, a full comparison
with a base branch uses roughly `(B+1)*S*H` simulator decisions per motion
realization, plus B cloned K-step neural updates. This cost must be charged to
training. The small target coordinate system and response products may make the
neural part cheap; they do not make IPC cheap. There is no current measurement
showing this solver saves wall time. A full-state/global-return HaDES search
would be an especially poor default for the present simulator throughput.

#### Why the joint choice can differ: an analytical example

This is an illustrative construction, not a dressing measurement. Consider a
policy with actions `a_1=theta` and `a_2=2 theta`, initialized at zero. State 1
rewards action 1; state 2 gives reward 1 at action 1 and 0.9 at action 2, with
narrow successful intervals around those actions. Independently selecting the
highest-scoring teacher action gives targets `(1,1)`. One SGD step with learning
rate 0.2 on `0.5[(theta-z_1)^2+(2theta-z_2)^2]` produces `theta=0.6`, actions
`(0.6,1.2)`, and zero return when the interval radii are 0.05.

Selecting targets `(1,2)` produces `theta=1`, actions `(1,2)`, and combined return
1.9. The lower-scoring second correction fits the shared policy better. HaDES,
cost-sensitive policy optimization and other existing methods can also address
such examples; it illustrates the mechanism, not novelty or superiority.

### Prior-art attack: where a contribution could and could not remain

The recursive search expanded from EXPO/DAgger to privileged teacher adaptation,
performance-based data curation, bilevel teaching, derivative-free optimization,
and constrained kernel policy updates. Broad slogans failed the novelty test:

| Closest primary source | Existing overlap | Narrow remaining question for this candidate |
|---|---|---|
| [EXPO](https://arxiv.org/html/2507.07986v3) | Value-guided edits followed by supervised absorption into a base policy | Can explicitly coupling correction choices through the finite base update reduce lost improvements per interaction budget? EXPO's RL/editor structure itself is not ours |
| [WISE](https://arxiv.org/html/2609.03681v1) | Scheduled counterfactual imagination, candidate feedback, policy post-training | Physics branches alone are not new; the proposed query concerns an updated policy, with joint target choices |
| [Guided Policy Search as Approximate Mirror Descent](https://proceedings.neurips.cc/paper_files/paper/2016/file/a00e5eb0973d24649a4a920fc53d9564-Paper.pdf) | Local improvement coupled to fitting a representable global policy | Local/global agreement and projection are established; the candidate needs a useful solver through the finite optimizer rather than a renamed agreement penalty |
| [Student-Informed Teacher Training](https://arxiv.org/html/2412.09149v2) | Teacher rewards and gradients account for student mismatch | The candidate optimizes joint corrections through a particular updated student's physical execution, not just action/KL agreement; student awareness is not new |
| [CUPID](https://cupid-curation.github.io/) | Uses influence estimates to rank existing demonstrations by closed-loop policy performance | Joint synthesis/selection of correction targets and direct branch feedback differ from independent data ranking; data influence itself is established |
| [Meta Pseudo Labels](https://openaccess.thecvf.com/content/CVPR2021/papers/Pham_Meta_Pseudo_Labels_CVPR_2021_paper.pdf) | Teacher targets adapt using the student's post-update supervised performance | Looking through a learner update is established; substituting a robotic task alone is insufficient |
| [Behaviour Distillation / HaDES](https://arxiv.org/html/2406.15042v1) | Synthetic state-action data optimized by the return of the policy trained on them; ES outer loop and fixed-initialization variant | The generic bilevel objective above is already covered. Only a demonstrably efficient local physical-query solver for ongoing post-training could distinguish our candidate |
| [Guided Evolutionary Strategies](https://proceedings.mlr.press/v97/maheswaranathan19a/maheswaranathan19a.pdf) | Surrogate gradients guide a low-dimensional black-box search | Sampling parameter directions from imitation gradients is not a contribution; compare with such direct search |
| [Constrained policy gradient using NTK](https://arxiv.org/abs/2107.09139) | Uses predicted cross-state policy changes and auxiliary training signals to enforce action-probability constraints | The response kernel and constraint synthesis are not new; the candidate must add useful physical-outcome-driven joint correction selection. This entry is based on the primary abstract |
| [CLIC](https://arxiv.org/html/2502.07645v3) | Supervision by desirable action sets | Keeping several acceptable actions is not by itself new; ordinary set-valued fitting is a necessary simpler comparator |

**Potential contribution, stated narrowly:** a practical post-training solver
that uses cross-state responses of a finite learner update to jointly allocate
physical correction targets, and spends limited reset-based interactions on the
behavior those targets actually induce. The claimed benefit would be fewer
lost corrections and fewer regressions across garment/motion contexts at equal
total computation. This search has **not established novelty for the complete
solver**. Its objective, response identity, trust regions and outer-loop search
all have precedents. A version equivalent to HaDES in another parameterization,
ordinary GPS/Q-loss, or gradient weighting would not justify a new-method claim.

### Development deliverables, without another baseline-evaluation campaign

1. **Specify the update operator and its interfaces.** The formulation above
   is the research deliverable now. The next implementation unit is a functional
   clone of the existing adapter/optimizer update, exposing its response to
   targets at a batch of real histories. Include no-update behavior, anchors
   and exact action scaling. Do not switch architectures at the same time.
2. **Implement joint target proposals and a training branch callback.** Separate
   the local physical surrogate from the learner response, and record every
   executed action/observation/return/termination. Existing held-action endpoint
   logs can seed candidate proposals, but cannot be silently relabeled as
   closed-loop updated-policy returns. M4 completion is not needed to develop
   these interfaces. None of this candidate is implemented yet.
3. **Use one bounded method-training comparison after implementation.** The
   scientific question is whether joint target selection retains more physical
   benefit than independent targets and simpler performance-based selection,
   with shared initial checkpoint, observation interface, task objective and
   total interaction/learning budget. This is a later proposal, not a new job or
   permission to append repeated M4 controller comparisons. Preserve a separate
   untouched garment/motion holdout for final reporting.

Cheap ablations remove the off-diagonal learning response, substitute a scalar
learning-rate search, or fit independent action sets using the same physical
data. Existing methods, particularly EXPO, GPS/Q-loss and performance-based data
selection, remain substantive alternatives. If the joint response has no useful
effect beyond those controls, or costs more than its gains justify, retire this
candidate's algorithm claim. If useful physical corrections are absent, it is
the wrong mechanism to pursue. Do not infer this condition just from r2's older
aggregate result, and do not claim an absorption failure before observing one.

Both research axes remain: moving arms supply temporal histories and changing
contact situations; ClothesNet supplies different geometries and opportunities
for beneficial or harmful transfer between corrections. Their inclusion in the
training objective does not guarantee unseen-category generalization. A broad
contact-rich post-training claim would ultimately require another task as well.

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

**Historical ordering:** the later method-development instruction above
supersedes the follow-up evaluation sequence in this section. Its evidence and
prior-art cautions remain applicable; its proposed additional evaluations are
not the active work queue.

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

Use the first-principles task framing at the top of this document. The joint
correction/update proposal is retired following the owner's rejection. Do not
add repeated old-controller evaluations or implement that proposal by default.
Preserve M4's already authorized budget and existing reporting. The dynamic
status above is the last inspected snapshot, not a new runtime check. This
literature/method revision changed no jobs and produced no new experimental result.

If later evidence justifies an EXPO comparison, its first implementation unit
is still a shared replay/outcome interface with complete successor states,
rewards, termination and consistent temporal semantics, followed by the editor
and critic. The current candidate logs are not Bellman replay. The metric and
relative-value extensions remain unimplemented comparison ideas; the new
literature findings do not justify presenting either as a discovered algorithm.
