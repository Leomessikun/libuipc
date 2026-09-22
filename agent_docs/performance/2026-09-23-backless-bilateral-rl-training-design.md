# Backless seated bilateral dressing: an RL training design

Date: 2026-09-23. Research and source inspection only. No tests, simulations,
training, or new empirical reanalysis were run.

## Decision and task contract

The owner corrected the chair specification: **there is no backrest**. The
current candidate is a seated person with both arms raised forward, a single
connected garment approached from behind, both hands already inside the
corresponding sleeve cuffs, and two robot grippers advancing the garment. One
sleeve may advance before the other. The objective remains the garment worn on
both shoulders and torso after release, not just two sleeves near the shoulders.
Garment type, grasp sites, arm-pose range, and the exact initial drape are still
design assumptions; a rear-opening gown is the simplest proposed first asset.
A pullover additionally needs a head/neck passage and is a different task.

Removing the backrest removes a major rear-access obstruction, not the need to
check robot reach, the chair seat and any armrests, head/torso clearance, the
garment's front/back orientation, and a collision-free route for the panel. Two
hands inside cuffs alone do not establish that the connected garment has the
correct topology around the body.

## What the current repository actually supplies

- `python/uipc_manip/dressing_env.py` has one 6-D kinematic grasp command, a
  right-arm-only rigid collider, one garment anchor patch, and a one-sleeve
  progress interface. It has no dual robot kinematics, chair, whole-torso
  contact, release action, or post-release completion metric.
- `python/uipc_manip/dressing_assets.py` loads one right cuff opening and grasp
  patch. Its cached states are not bilateral transition data. The full body mesh
  is generated in `dressing_body.py`, but the active collider is extracted from
  the right arm.
- `python/uipc_manip/dressing_reward.py` scores one sleeve's arm progress and
  alignment. It does not score whole-garment completion, physical grasp
  retention, distributed body contact load, or cloth damage.
- The legacy actor observation is a one-arm point cloud and one tool point.
  Arm contact-force readout is off by default and historical transient model
  readings are not validated human-force measurements. True sleeve progress and
  the rear panel position are simulator labels, not automatically available to
  the deployed actor.
- The existing simulator returns to one settled initial snapshot. A curriculum
  of physically valid partial-wear states and pose/drape variants needs a new
  reset and scene-generation path. The current single-arm branches cannot be
  relabeled as dual-arm experience.

This is therefore a new task environment, not an alternative trainer command
for the existing one.

## Research question and candidate method (not a novelty claim)

The baseline below is a way to make the task trainable, **not the research
contribution**. The candidate question is whether RL can learn the *effect of
one gripper on the other sleeve and the torso* and use that information to
choose a joint action that preserves the possibility of finishing the whole
garment. A left pull can advance the left sleeve while tightening or displacing
the right sleeve or panel; the relevant outcome is after subsequent control and
release, not the left sleeve's immediate progress. This mechanical coupling is
specific to one connected garment, but the idea of counterfactual multi-agent
credit assignment is not new: [COMA](https://ojs.aaai.org/index.php/AAAI/article/view/11794)
already uses a central critic with per-agent counterfactuals. Likewise,
[PA-BiCoop](https://arxiv.org/pdf/2606.28192) already learns dynamic arm roles
from demonstrations, and [Wearing A Coat](https://arxiv.org/html/2607.10999)
already handles sleeve coupling with a staged cloth-aware MPC. A role-switch
head, joint critic, or ordinary cross-action feature alone would duplicate
existing ideas.

One candidate **learning operator**, conditional on the scene and baseline
working, is a budgeted *coupled-consequence query*:

1. At a physically reachable checkpoint where a joint policy is uncertain
   about advancing left, right, or both, form four executable short commands
   from the **same** checkpoint: joint $(u_L,u_R)$, left-only $(u_L,0)$, and
   right-only $(0,u_R)$, plus a hold branch $(0,0)$ to estimate a pure
   interaction term. These are controlled counterfactuals, not expert labels.
   Match initial conditions and solver randomness where possible; repeat only
   where different outcomes would change the preferred action.
2. Record a vector of **valid** left progress, valid right progress, torso
   progress, loss-of-threading/grasp events, and eventual post-release success.
   A short branch can label immediate coupling; it cannot certify eventual
   success. Continue selected branches under the *same current policy* until
   release, and recheck when that continuation policy changes. For any outcome
   component $Y$, the interaction contrast is
   $Y(u_L,u_R)-Y(u_L,0)-Y(0,u_R)+Y(0,0)$; this contrast is a diagnostic of
   joint mechanical effects, not a new gradient identity.
3. Fit a critic/outcome model to ordinary RL transitions **and** these paired
   outcomes. The difference between joint and unilateral outcomes estimates
   whether the two pulls help or hinder each other. Use the learned model to
   select the next *feasible joint action* and to allocate another physical
   query only where uncertainty could reverse that choice. Update the actor
   from actually executed returns; do not treat an unverified model prediction
   as a success or imitate a privileged teacher trajectory.

This is a **hypothesis for a new operator**, not established novelty or an
implemented algorithm. Its distinctive test is whether *bilateral mechanical
externalities and eventual completion* justify the extra matched queries.
Compare under equal total workstation time against the plain joint RL learner,
fixed left-first/right-first/synchronous pulls, uniform counterfactual query
allocation, a COMA-style counterfactual critic, and the applicable staged
cloth-aware controller. Measure executed whole-garment success after release,
not surrogate ranking on queried snippets. If the plain joint learner or a
fixed order matches it, or the query cost erases its gain, the research claim
fails. Do not turn this proposal into a training launch without first making
the bilateral task physically coherent and measurable.

## First trainable policy (baseline)

Start with one rear-opening gown, one body-size range, fixed prethreaded cuffs,
two fixed garment grasp sites near the shoulder/upper-sleeve regions, and static
human arm poses during each episode. Simulate both robot arms or enforce their
reachable end-effector sets and link collisions through a robot model. Preserve
one connected cloth mesh and all relevant body/seat contacts. A bounded
Cartesian/impedance controller maps policy targets to reachable joint motions
and limits speed, robot collision, and measured gripper load. It does not move
the human or freely reposition cloth vertices.

**Policy observation:** both robot/gripper states, estimated left/right human
arm frames, partial garment point clouds or tracked landmarks from views that
actually cover the rear task, gripper load measurements, and a short action/
observation history. The actor must not receive exact simulator mesh positions,
hidden sleeve-root progress, or body contact force unless equivalent deployable
sensing is established. The actor may receive estimated garment progress with
uncertainty. A privileged critic may use full simulation state during training;
this is asymmetric actor-critic RL, not a teacher policy or distillation.

**Policy action:** at each feedback decision, output two bounded 3-D gripper
displacements in the corresponding human-arm frames. Equivalently, represent
them as a common pull plus a left/right differential and lateral corrections;
the representation does not remove any physical degree of freedom. Track
gripper orientation with the low-level controller initially; add a learned
rotation only if the garment's actual motion requires it. A single joint actor
can choose unilateral motion by moving one gripper little or not at all, and
can choose simultaneous motion by moving both. No explicit left/right/both
mode is necessary for this first task.

**Learner:** use an off-policy twin-critic continuous-control baseline such as
TD3, with transitions from executed physics, a centralized Q-function over
both robot actions, and an actor that sees deployable observations. A replay
buffer reuses every expensive physical transition for network updates. If the
two-side partial observation is aliased, add history rather than giving the
deployed actor hidden simulator labels. Neither simulator differentiation nor
the old SAC pretraining pipeline is required. TD3 and asymmetric critics are
established tools, not claimed algorithmic contributions.

## Training signal and curriculum

For each sleeve, compute a **valid** progress scalar from the proximal sleeve
root along the corresponding hand-to-shoulder arm path, separately checking
that the correct hand remains inside the sleeve and that the cuff stays near
its intended distal position. Let `p_L`, `p_R`, and `p_T` denote left and right
valid progress and torso/shoulder placement. Dense shaping can use the change
in a potential combining mean bilateral progress, the slower side's progress,
and torso progress. The minimum term makes sacrificing the lagging sleeve
unattractive; the mean term still rewards a valid unilateral advance. The
terminal bonus is paid only after both shoulders and torso satisfy the task
geometry and the garment remains worn following controlled gripper release and
settling. Failure events include lost threading, actual loss of a required
grasp, cloth damage, invalid robot collision, and calibrated load violations.
Keep these event flags through the whole episode. The legacy 2 cm anchor error
is a proxy, not by itself a real-gripper or human-safety certificate.

Training starts with both wrists prethreaded and a narrow forward-raised pose
distribution. Generate valid initial drapes by settling the connected garment
with both hands in the correct sleeves; reject self-intersection and wrong-side
panel configurations. Use valid states reached by actual execution as optional
partial-wear resets, including left-ahead, right-ahead, and balanced cases;
never create an easy reset by teleporting the sleeve along the arm. Keep
original wrist-start episodes in the mixture throughout training. Then widen
the independent left/right pose ranges, initial drape, body shape, and garment
parameters in stages. Only include configurations with feasible robot reach
and garment fit. Hold out complete pose/body/garment combinations for final
assessment.

Hindsight Experience Replay is **not a default component**. It only applies if
the actor is explicitly goal-conditioned and an actually achieved bilateral
goal can be relabeled without changing immutable slip, collision, or load
failures. It cannot turn a rollout with no complete dressing into complete
task success. A high-level SMDP selector is also optional: introduce it only
if a continuous joint actor cannot express required regrasp or discrete
support-transfer choices, or if different mode orders demonstrably change
success. Its variable-duration returns would then need correct elapsed-time
discounting and physical termination conditions. Both additions increase
implementation and learning complexity.

For this expensive scene, terminate episodes that have irreversibly lost a
required sleeve/grasp, reuse replay data, and report total simulator steps and
wall-clock time. The old single-arm throughput and earlier training estimates
cannot predict the cost of full-body bilateral contact. A lower-resolution or
cheaper simulator is useful only if it preserves the action consequences that
determine the policy; the training design is not tied to IPC.

## Research position and checks for a final claim

[One Policy to Dress Them All](https://roboticsproceedings.org/rss19/p008.pdf)
already learns across arm poses and reports preliminary fixed-pose dual-arm
simulation. [Wearing A Coat](https://arxiv.org/html/2607.10999) already controls
both sleeves through sequential and simultaneous phases, including seated
human experiments with compliant human motion. Thus a backless chair, bilateral
pulling, body-relative actions, TD3, curriculum, or asymmetric critic alone is
not a new RL algorithm. The possible research question is whether one policy
can select and execute valid bilateral progress across varied postures and
partial-wear states more effectively, at matched physics cost, than fixed
left-first, right-first, synchronized pull, a comparable flat RL policy, and
an applicable cloth-aware planner/controller. If a fixed order works equally
well, an explicit ordering learner has no demonstrated value.

The first deliverable is a physically coherent task environment and a simple
joint-policy baseline with observable inputs and a full-garment endpoint. The
research deliverable would be a measured improvement in *executed* completion
per total cost attributable to the coupled-consequence operator, beyond the
counterfactual and planning precedents. No training-hour estimate, algorithmic
novelty, or success claim follows from the present code and literature review.
