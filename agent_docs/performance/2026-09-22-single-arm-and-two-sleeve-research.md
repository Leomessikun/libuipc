# Dressing research: one human arm first, both sleeves as an alternative

**Task priority superseded by the owner's next instruction.** The owner selected
both-arm and torso dressing for a seated person with both arms raised forward,
and explicitly allowed a non-RL approach. The single-arm recommendation below
is historical, not the current plan. See the
[seated dressing implementation review](2026-09-22-seated-bilateral-dressing-plan.md).

Date: 2026-09-22. Literature review and reasoning only. No tests, simulation,
training, numerical probes or new empirical analysis were performed.

## Decision and corrected scope

Retain **one-human-arm dressing** as the first research scope. There is a
defensible question about inserting a moving hand into a deforming sleeve
opening. Expanding to both sleeves is unnecessary merely because the previous
proposal was unconvincing. Two-sleeve dressing is a separate alternative with
its own established literature and additional physical coupling.

The owner's distinction is explicit:

| Task | Meaning |
|---|---|
| Single-arm dressing | Put one sleeve onto one human arm. |
| Two-arm dressing | Put the two sleeves of the same garment onto both human arms. |
| Two robot manipulators | A hardware choice that can occur in either task. |

The earlier [support-transfer recommendation](2026-09-22-bimanual-dressing-research.md)
confused the last row with the second. It is withdrawn. Neither that task nor
its TD3 reference learner was selected by the owner.

This review recommends a research question, not a finished new RL algorithm.
It specifies a candidate learning mechanism and the evidence it would need.
Teacher/student learning, distillation and DAgger remain excluded. The old
Stage 0–3 schedule remains retired; IPC and SAC are not requirements.

## What the literature actually establishes

Primary sources were checked beyond the previously supplied
[Lin Shao publication page](https://linsats.github.io/). A limitation in a recent
paper is useful evidence, but does not prove that all earlier work shares it.

| Primary source | Relevant capability and boundary |
|---|---|
| [Dressing in Motion, September 2026 preprint](https://arxiv.org/html/2609.04759) | Its problem definition holds the arm stationary before insertion, then permits motion. It does not require initial sleeve–hand alignment. Its limitations explicitly discuss pre-insertion movement disrupting alignment. This supports investigating that phase, not claiming all moving-arm dressing is new. |
| [Force-Modulated Visual Policy, CoRL 2025](https://arxiv.org/html/2509.12741) | Adapts dressing to human motion using vision and force. Its limitations describe motion primarily after partial insertion; early inward rotation and lowering can cause difficult failures. This is a qualified operating condition, not a universal ban on earlier motion. |
| [Graph Dynamics MPC for Garment Opening Insertion, ICRA 2024](https://www.imperial.ac.uk/personal-robotics/publications/?id=1405147&noscript=noscript&respub-t4-action=citation.html) | Already learns garment-opening dynamics for MPC in the presence of a body. The institution's abstract was inspected; unavailable full text prevents a claim that every dynamic-entry condition was excluded. |
| [Garment Diffusion Models, RA-L 2025; online 2024](https://spiral.imperial.ac.uk/entities/publication/a0176f43-5eb7-4530-93d8-35c0118514df) | Predicts future opening geometry from partial observations and actions, and iteratively trains an MPC system using model-based RL. Learning an opening model and planning with it is already a direct dressing precedent. |
| [Cognitive overloading and distractions during dressing, 2022](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.815871/full) | Studies disrupted human movement, including the phase before physical coupling. The relevance of unexpected pre-contact movement was identified before the recent diffusion-policy literature. |
| [Initial dressing motion while walking, 2024](https://www.jstage.jst.go.jp/article/jsmermd/2024/0/2024_2A2-L08/_article/-char/ja/) | The authors' conference abstract describes trajectory optimization and feedback for initial assistance with walking and arm swing. The related CASE paper, [Motion Generation for Mobile Manipulators to Assist in Putting on a Jacket While Walking](https://doi.org/10.1109/CASE59546.2024.10711375), was identified, but its full text was not retrieved. Do not claim the first pre-contact assistance to a moving person. |
| [Synchronous dressing support based on working-status prediction, 2026](https://link.springer.com/article/10.1186/s40648-026-00350-9) | Already predicts movement and force for MPC. The person uses the able arm to manipulate the garment while the robot assists sleeve pulling. This differs from autonomous initial alignment of an unsupported moving hand, but rules out claiming motion prediction plus dressing MPC as new. |
| [TOORAD, 2019](https://link.springer.com/article/10.1007/s10514-019-09865-0) | Personalizes collaborative plans to geometry and mobility. Candidate subtask sequences are supplied manually and include dressing each forearm before the upper arms. Experiments include assistance with both sleeves. Neither arm ordering nor partial progress on both sides is an empty field. |
| [Learning garment manipulation policies toward robot-assisted dressing, 2022](https://pubmed.ncbi.nlm.nih.gov/35385294/) and [author implementation](https://github.com/fan6zh/robot_dressing) | Demonstrates a garment preparation and dressing pipeline on a medical manikin, including sequential sleeve dressing. A complete two-sleeve task is not itself a first demonstration. |
| [Wearing A Coat, July 2026 preprint](https://arxiv.org/html/2607.10999) | Explicitly models the connected garment. Its global controller limits first-side upward progress and moves that opening along the arm toward the remaining hand when needed. Once both arms are inserted it advances both sides. The first sleeve restricting the second is already handled explicitly. |

The defensible conclusion is narrower than either “single-arm dressing is
solved” or “nobody handles motion before entry.” Existing capabilities leave
room to investigate **reliable autonomous entry under uncertain hand motion and
action-dependent opening deformation**. Its benefit over predictive control is
unestablished.

## First direction: learn which moving sleeve–hand encounters permit entry

### The physical question

A robot approaches a hand with a sleeve. During the approach, the person raises
the hand or rotates the wrist. Moving toward the last observed hand position
can arrive too late. Moving toward a predicted hand position can also fail:
the robot's movement may narrow or rotate the sleeve opening before arrival.

The desired capability is to choose movements that create an adequately open,
properly oriented passage **when the hand reaches it**, then retain the insertion
and continue dressing. Sometimes the useful action is to change approach angle;
sometimes it is to wait briefly, widen the opening, or withdraw and approach
again. These are illustrative choices, not measured outcomes or a mandatory
hand-coded action sequence.

The question is therefore about the joint future of the hand and the opening.
Tracking the hand alone does not answer it. A rigid ring centered at the right
position is also an inadequate model of an opening that folds under pulling.

For example, two plausible future hand paths might pass on opposite sides of
their mean prediction. Aiming at that mean could miss both. A useful controller
could instead maintain an opening configuration that accommodates a range of
arrival positions and times, or wait for new observations to resolve the choice.
This example motivates a distribution over entry opportunities; it does not
claim that probabilistic prediction or uncertainty-aware control is new.

### Proposed learning mechanism

Investigate a representation of **entry opportunities in geometry and time**.
The representation would preserve the observed opening boundary, its orientation
and deformation, hand pose and recent motion, robot configuration, and relevant
force/history information. A centroid alone loses whether the passage is open
and on the correct side of the hand. Hidden geometry remains uncertain rather
than being treated as an observed complete mesh.

The central learned quantity would estimate the probability of attaining and
retaining valid insertion within a remaining time budget, conditional on the
current observation history and robot action. One precise reference quantity is

$$
P^\pi_\tau(h,a)
=\Pr(T_{\mathrm{entry+hold}}\leq\tau,
      T_{\mathrm{entry+hold}}<T_{\mathrm{failure}}\mid h,a,\pi).
$$

Here, $h$ is the sensor/action history, $a$ is the next robot command, $\pi$ is
the subsequent policy, and $\tau$ is the remaining decision budget.
$T_{\mathrm{entry+hold}}$ is the first time that correct insertion and a specified
retention interval have both been achieved; $T_{\mathrm{failure}}$ is the first
specified terminal failure. This is a learned probability, not a certificate.

The proposal is to use opening geometry and human-motion uncertainty to learn
this quantity across different entry encounters, and connect it to the value
of completing the rest of the sleeve. Favoring easy entry that leaves the cloth
in an unusable configuration would not satisfy the task. The research question
is whether this structure improves learning and transfer over a generic value
network with the same observations and budget.

This differs from the abandoned goal-imitation stage: the robot learns from
executed consequences. No teacher supplies the correct opening target, and no
student copies a geometric controller.

### How training and deployment would work

1. Collect ordinary interaction transitions containing observations, actions,
   successor observations, entry/retention events and termination information.
   Simulator state can provide training labels; the deployed policy receives
   sensor history. Moving-arm experience must actually be represented in this
   data. Existing static-arm branches cannot supply it by assumption.
2. Train a finite-horizon action-value learner from replay, with the remaining
   deadline and entry/retention phase represented. Nonterminal targets use the
   successor value at a shorter horizon. The actor improves against the learned
   value and task costs. This is off-policy RL, not behavioral cloning.
3. Use complete dressing outcomes to evaluate and learn continuation after
   entry. A local crossing or coverage peak is insufficient. Replay collected
   under an older continuation must be updated through appropriate value
   targets; its outcome is not an immutable label for a candidate first action.
4. At deployment, process current sensor history and issue the next feasible
   robot command, then update after new feedback. The proposed actor does not
   require online IPC branches. Robot kinematics and low-level feedback remain
   necessary, and end-to-end perception/control latency must be accounted for.

A finite deadline makes endless waiting unsuccessful. Human motion would need
a stated operating range; no predictor can guarantee an arbitrary future change
of mind or posture. These assumptions define the task rather than implying
that all human movement is predictable.

### What could be new, and what is already known

Time-conditioned value learning is established by
[Temporal Difference Models, ICLR 2018](https://arxiv.org/abs/1802.09081).
Reach–avoid objectives and actor–critic updates are also established machinery.
Neither the probability equation nor adding a temporal output head establishes
a new algorithm. Garment dynamics prediction already has the direct precedents
in the table.

The candidate method contribution is the way a learned representation of the
deforming passage and uncertain relative motion supports transferable entry
decisions and subsequent completion. It must explain a benefit beyond more
history, a better motion predictor, extra parameters or more simulation.

The close comparisons would therefore be predictive hand tracking, garment
dynamics MPC with the same motion information, and an ordinary temporal
actor–critic with identical sensing and experience. All would need the same
pre-insertion motion conditions. A favorable comparison only against a policy
whose task definition requires a stationary hand would be insufficient.

This is currently a substantive **capability and learning-method hypothesis**.
There is not yet a justified claim of a new general RL update rule. If ordinary
predictive MPC solves the problem comparably, the proposed representation does
not earn an algorithmic claim merely because it is used for dressing.

## Alternative: learn how to allocate progress between two connected sleeves

This alternative means both human arms entering the same garment. An open-front
garment would isolate sleeve coordination from head insertion and fastening;
that is a possible task definition, not an accepted hardware or garment choice.

The physical question is how to distribute progress when improving one sleeve
changes what remains achievable on the other. For example, lifting the first
sleeve farther may make it harder to bring the second opening to a bent arm.
Different garment widths and available human joint motion can change when to
switch sides, retain partial progress, or reverse part of an earlier movement.

That observation alone is already in the literature. In particular, Wearing A
Coat uses explicit stage-dependent objectives and garment-distance conditions;
its mannequin example includes a preset partial-progress threshold. TOORAD also
includes alternative manually specified partial-dressing sequences. The proposed
difference must concern adaptation beyond these mechanisms.

A candidate learning object is the **joint set of attainable progress pairs**:
how much progress can be achieved on the left and right together from the
current garment/body state, under the available robot and human motion. This
could inform which side to advance next and how far, including cases where
temporary retreat preserves the possibility of finishing both sides.

Training would require transitions from a complete connected garment and a
joint completion objective. A policy could learn direct robot control together
with goal-conditioned values for progress pairs, updated from executed outcomes.
Independent left- and right-sleeve successes cannot simply be combined into a
successful joint episode. Nor can single-arm trajectories be treated as
independent transition samples once the same garment transmits tension between
the two sides.

Hierarchical RL, goal-conditioned values and a reward for finishing both arms
are not new by themselves. The research hypothesis concerns learning the
state-dependent loss of future options caused by current sleeve progress, and
using it to improve control across garments and mobility restrictions. A joint
critic alone could already learn this; an explicit progress-set representation
must justify its additional structure against that baseline and the existing
global garment planner.

Deployment would require the controller to observe both sides and garment
coupling. If the task permits human assistance, the permitted movement must be
explicit; a robot policy cannot be credited with commanding a person's joints.
Two sleeves add longer credit assignment, more occlusion and coupled contact.
There is no evidence that this alternative is cheaper or easier to train.

## Scaling claims that survive the literature check

The direct garment-diffusion paper reports 91.2% task success using fewer than
100 sampled trajectories in its insertion setting. Its institution's record
does not establish an all-inclusive workstation-hour budget for our setting.
This is useful evidence that model-based dressing learning can work, not a
promise that our proposed dynamic-entry learner will train cheaply.

Other tempting shortcuts also require qualification:

- [Task-Driven Hybrid Model Reduction](https://arxiv.org/abs/2211.16657) already
  studies learning a compact task-relevant contact model for control. Its very
  small data/time examples concern simulated rigid-object manipulation, not a
  measured cloth-dressing budget. A small model plus MPC is not a new idea.
- [CoDA](https://arxiv.org/abs/2007.02863) and
  [MoCoDA](https://arxiv.org/abs/2210.11287) already exploit local dynamical
  independence for RL data augmentation. Recombining hand and garment motion
  before contact is therefore not automatically new. It is also invalid when
  their dynamics or the action/history being reused are coupled. Predicting
  that contact will occur does not label the post-contact dressing outcome.
- [Disentangling perception and reasoning for cloth learning, 2026 preprint](https://arxiv.org/html/2601.21713)
  uses cross-modality Q distillation from a full-state agent and reports about
  40 hours on an RTX A6000 across its training stages. Its demonstration-free
  framing does not make it compatible with the owner's no-teacher/student rule,
  nor does it establish a short dressing-training schedule.

For the proposed single-arm question, shorter entry episodes and replay reuse
are plausible ways to limit interaction cost. They do not make each physical
decision cheaper. The eventual budget must include interaction collection,
perception/model training, policy updates, resets and complete-task evaluation.
An uncertainty-aware model could be used for additional experience only where
its errors are controlled; it cannot replace missing contact evidence by decree.
Interaction collection would need a simulator suitable for the task and its
budget; no requirement ties it to IPC. Existing static-arm throughput cannot
be assumed for moving-body or two-sleeve contact. No two-day estimate or fixed
speedup is supported by this review.

## Research priority

Investigate the single-arm entry question first because it is concrete and
supported by identifiable limitations, while preserving most of the task scope.
Keep learned coordination of both sleeves as an alternative, not an automatic
expansion. The more demanding two-sleeve task does not remove the need to explain
what the learning method contributes over its strong planning precedents.

The literature rules out several superficial novelty claims and identifies
where a method could make a difference. It does not yet establish that either
candidate beats its nearest baseline. This document changes the research
assessment and corrects the task interpretation; it authorizes no execution.
