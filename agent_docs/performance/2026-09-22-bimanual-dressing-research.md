# Research reset: dressing with changing grasps and supports

**Recommendation withdrawn after the owner's scope correction.** Two-arm
dressing meant both human arms in one garment, considered only as an alternative
to a substantive single-arm research direction. This report instead recommended
two robot manipulators dressing one human arm. That was the assistant's
misinterpretation, not the owner's selected task. Its literature notes remain
historical; its learner and hardware assumptions are not an execution plan.
See the [corrected review](2026-09-22-single-arm-and-two-sleeve-research.md).

Date: 2026-09-22. Primary-source literature review and existing-document
inspection only. No tests, simulation, training, numerical probes, or new
empirical analysis. The owner permits changing the task, including bimanual
dressing, and continues to exclude the teacher/student route.

## Recommendation and confidence

Investigate **bimanual recovery during dressing: change grasp/support while
preserving a useful garment–body insertion relationship, then finish dressing**.
Initially, two robot arms manipulate one sleeve around one stationary mannequin
arm. This is distinct from promising a complete two-sleeve coat pipeline.

This is a recommended research problem, not a validated new RL algorithm.
Adding a second arm, a restabilization policy, long-term grasp values, or TD3 is
already covered by prior work. The specific hypothesis is that learning how
support changes affect insertion retention and subsequent manipulability can
enable recovery and transfer across grasp configurations more effectively than
direct action-value learning. The method contribution remains conditional on
demonstrating that difference. A literature search cannot establish its benefit
or certify that no equivalent method exists.

The previous action-order estimator is **not the selected route**. It has no
established dressing bottleneck, completion-learning signal, or total training
cost advantage. The old Stage 0–3 pipeline also remains retired. We retain its
objective corrections and measured evidence, rather than its training schedule.

Hardware availability has not been confirmed. The design below assumes
simulation research first; a real bimanual dressing capability claim requires
an actual bimanual platform. No implementation or run is launched by this note.

## What Lin Shao's page adds

The starting point was the owner's previously linked
[publication page](https://linsats.github.io/). The useful lesson is to choose
the physical decision that learning should solve. The following works also
place concrete limits on possible novelty claims:

| Primary source | Relevant contribution and boundary |
|---|---|
| [Learning to Regrasp by Learning to Place, CoRL 2021](https://proceedings.mlr.press/v164/cheng22a.html) | Learns placements that enable regrasping, including creating supports using other objects. Changing the environment to make a subsequent grasp possible is established. Its rigid-object placement setting differs from cloth already constrained by a body. |
| [Bi-Adapt, 2026](https://arxiv.org/html/2602.08425) | Uses semantic correspondence and interaction feedback to adapt bimanual affordances across articulated-object categories. Sequential conditioning of the two actions is established. Its 50-interaction adaptation setting does not include the cost of pretraining; the reported long-horizon limitation matters here. |
| [ManiFoundation, 2024](https://arxiv.org/html/2405.06964) | Predicts contacts and forces/motions given desired object motion, including deformables. It assumes that desired motion is provided. It does not supply a complete dressing strategy or establish that an arbitrary desired sleeve motion is achievable with the current grasp. |
| [ContactExplorer, 2026 preprint](https://arxiv.org/html/2603.10971) | Uses PPO with state-conditioned contact coverage and approach rewards. The learning object is structured exploration. Contact novelty does not itself distinguish helpful cloth sliding from tightening a jam. |
| [D(R,O) Grasp, 2024 preprint](https://arxiv.org/html/2410.01702) | Represents robot–object distance relations and reconstructs executable grasps. Relational outputs have precedent; distances alone do not establish sleeve insertion or hidden cloth-layer relationships. |
| [TieBot, 2024](https://arxiv.org/html/2407.03245) | Structures knotting through reconstructed cloth subgoals and grasp decisions. Its teacher/student recipe is excluded by the owner; its task decomposition is useful background, not our proposed training pipeline. |
| [AdaptPNP, 2025 preprint](https://arxiv.org/html/2511.11052) | Combines manipulation primitives with digital-twin rehearsal and feedback. Its rigid-body pose formulation does not provide the missing cloth contact representation. |

These papers do not justify assembling their components and calling the result
a new RL algorithm. The additional literature below is necessary because some
of the seemingly promising combinations already have direct predecessors.

## Closest work outside that page

| Primary source | What it rules out as a standalone claim |
|---|---|
| [Learning garment manipulation policies toward robot-assisted dressing, Science Robotics 2022](https://pubmed.ncbi.nlm.nih.gov/35385294/) and [author code](https://github.com/fan6zh/robot_dressing) | A dual-arm hospital-gown pipeline already includes garment preparation and sequential arm dressing on a medical manikin. Neither complete preparation nor bimanual dressing is an empty field. |
| [Bimanual Robot-Assisted Dressing: A Spherical Coordinate-Based Strategy for Tight-Fitting Garments, 2025](https://arxiv.org/html/2508.12274) | Two robot arms dress one human arm using demonstration-derived geometric coordination. That task configuration and structured coordination are established. |
| [Wearing A Coat, July 2026 preprint](https://arxiv.org/html/2607.10999) | Combines differentiable cloth simulation, global planning and local control for two-sleeve coat dressing. The reviewed method models garment attachments at the grippers. It is a close dressing-control comparison, not evidence that on-body regrasp learning has been solved. |
| [Stabilize to Act, CoRL 2023](https://arxiv.org/abs/2309.01087) | Already learns where to stabilize, when to restabilize, and an acting policy from demonstrations; jacket zipping is among its tasks. A support hand plus an acting hand, even with restabilization, is not our novelty. |
| [PA-BiCoop, June 2026 preprint](https://arxiv.org/html/2606.28192) | Explicit primary/auxiliary coordination and dynamic role assignment are established. Swapping left/right roles is insufficient novelty. |
| [HACMan++, RSS 2024](https://arxiv.org/html/2407.08585) | Uses spatially grounded primitive selection and continuous parameters with TD3. Learning where, which primitive, and how to act is established. Its reported tasks do not validate dressing or our proposed computation budget. |
| [Learning Foresightful Dense Visual Affordance, 2023](https://arxiv.org/html/2303.11057) | Already uses downstream state values for deformable-object action selection, with staged learning and interaction data. Merely scoring grasps by future progress is not new. |
| [Learning to Manipulate Deformable Objects without Demonstrations, 2020](https://arxiv.org/abs/1910.13439) | Demonstration-free visual learning and conditional pick/place decisions have direct precedent. Avoid a first-without-demonstrations claim. |
| [Dressing with topology coordinates, 2013](https://www.tandfonline.com/doi/abs/10.1080/01691864.2013.777012) | Low-dimensional topological relationships in dressing RL are established. Adding an insertion indicator is not a new representation by itself. |
| [AutoBag, 2023](https://arxiv.org/abs/2210.17217) | Opening a flexible bag and inserting objects using learned perception and manipulation primitives already exists. Moving from sleeves to bags is not sufficient novelty. |

The defensible gap suggested by this comparison is narrower than "regrasping
cloth": **learning support changes during body-constrained insertion, including
when temporary withdrawal helps, while distinguishing useful insertion from
visual overlap**. This is an inference from the reviewed methods, not an
exhaustive first-ever assertion.

## Three concrete candidates

| Priority | Task and method hypothesis | Main uncertainty |
|---|---|---|
| 1 | Recover a partly donned sleeve using two grippers. Learn support-conditioned insertion retention and continuation value; use them to choose pull, unload, acquire support, and regrasp. | Whether an explicit representation of support transfer contributes beyond HACMan++-style direct Q learning and ordinary constrained RL. |
| 2 | Learn a common opening-traversal skill for a sleeve, a sock-like tube, and a fabric cover over a bent fixture. Condition decisions on opening shape/orientation, insertion relation and available grasps. | Whether this representation transfers across topology and geometry better than goal point clouds or object-motion representations; one centroid is insufficient. |
| 3 | Resolve hidden contact ambiguity through a short bimanual interaction before choosing recovery. Learn probes for the value of their feedback to the next decision. | Whether sensing actually changes the best action, and whether benefit comes from information rather than the probe physically freeing the cloth. |

Select candidate 1 as the research focus. Candidate 2 is a potential later
generality study, not a second training project. Candidate 3 requires a stronger
observational ambiguity case and suitable force/tactile observations; it is not
justified by our existing macro results. Its value-of-information principle is
established decision theory, not a proposed new theorem.

### The initial task for candidate 1

Start with a sleeve opening already located around the hand or forearm, with
varied robot grasp positions and cloth slack. Two arms can grasp, release, move,
and exchange support. Complete passage over a bent elbow toward a specified
upper-arm region. The initial task does not include garment pickup from a pile,
both human arms, autonomous human-arm lifting, or moving people.

One example: continued pulling bunches fabric at the elbow. The robot can unload
the fabric, use one gripper to maintain the opening, relocate the other grasp,
and continue. This is an illustrative desired behavior, not a scripted policy
or an assertion that every jam is recoverable. Full withdrawal and reinsertion
may be necessary in some configurations; the task must not forbid such recovery
merely to make a preservation claim look stronger.

The scientific comparison is between controllers with the **same two arms,
release/regrasp actions, observations and budget**. Beating a single-arm agent
would establish an actuation advantage, not a learning-method advantage.

## A concrete learning design, with novelty separated from infrastructure

### What the policy observes and chooses

Train one recurrent policy directly on deployment-available observations:
partial RGB-D/point clouds, robot poses and gripper status, and measured wrist
forces if the platform supplies them. Retain observation/action history through
occlusion. Simulator topology and contact ground truth can produce training
labels and rewards; they are not unannounced inputs to the deployed actor.

Represent visible garment patches, the opening, body landmarks and current
supports as interacting features. Separate the task relationship (which part of
the limb is inside the sleeve) from the current grasp configuration. Changing
grasps can preserve the task relationship while changing the dynamics. Neither
an exact material correspondence nor full hidden-mesh reconstruction is assumed
available from a depth image.

The action is a parameterized manipulation command:

$$
a=(k,\ell,p,u).
$$

Here $k$ is a primitive such as acquire grasp, release, move, or coordinated
move; $\ell$ identifies the acting arm(s); $p$ is an observed garment point when
needed; and $u$ contains bounded motion/compliance parameters. A short feedback
controller executes the command and returns new observations. The policy chooses
the next command from those observations. The controller implements grasping,
IK and motion execution; it does not encode a successful dressing trajectory.

Candidate actions must include maintaining an existing grasp and simultaneous
two-hand movement. A permanently stationary support arm is too restrictive.
Only genuine kinematic/command impossibilities are masked. An uncertain learned
feasibility prediction is not treated as a certificate permitting irreversible
candidate elimination.

### Which RL algorithm, exactly

Use **spatially grounded, off-policy actor–critic learning following the TD3
design of HACMan++** as the reference learner. A network scores point/primitive
choices; an actor supplies continuous parameters. Experience replay trains twin
critics through Bellman targets and trains the actor through the critic. This is
RL, not supervised imitation of successful teachers, and it does not require
physics derivatives. TD3 is an existing algorithm, not the claimed contribution.

For a primitive lasting $\tau$ physical decisions, the ordinary target is

$$
y=\sum_{j=0}^{\tau-1}\gamma^j r_{t+j}
  +\gamma^\tau(1-d)V_{\mathrm{target}}(h_{t+\tau}).
$$

Here $h$ is observed history, $d$ denotes a true task terminal, and $V$ is obtained
from target critics and candidate actions. These are standard temporal-abstraction
semantics. Store terminal causes separately: success, task failure, time limit,
and simulator failure are not interchangeable labels.

### What would actually be researched

The proposed extension is **a representation and learning objective for support
transfer**, evaluated against the same learner without that extension:

1. From executed acquisition/release/movement transitions, learn whether the
   existing insertion relation survives, and which garment–body relationship
   changes occurred. Retain labels for failed transfers as well as successes.
2. Condition this prediction on the support configuration and observed history,
   so that superficially similar shapes with different held patches are not
   collapsed into the same control state.
3. Train continuation values on subsequent transitions and complete outcomes.
   A retained grasp, or an immediately successful transfer, is not labeled a
   successful dressing episode. The RL objective must reward eventual completion
   and charge meaningful task failures, rather than reward preserving a grip
   indefinitely.
4. Use shared task-relation features across grasp configurations, while retaining
   support-dependent features for action consequences. The hypothesis is better
   data reuse across grasps and garments, not that all grasps are dynamically
   equivalent.

This specifies a buildable hypothesis, but adding auxiliary predictions may
prove to be only an engineering improvement. To constitute a method result,
the representation must improve recovery or transfer at matched interaction and
total-time budgets over direct Q learning, downstream-affordance learning and
appropriate support/acting baselines. If the standard learner performs equally
well, there is no demonstrated new RL mechanism. A new capability or benchmark
could still be useful, but must be described as such.

### Data, rewards and scaling

Collect ordinary closed-loop transitions from the policy and bounded primitive
exploration. Store complete successor observations, grasp events, durations and
episode outcomes. Train the same shared policy across garments and grasp
configurations using replay. Physically settled, procedurally specified initial
states can define a curriculum; producing them is part of the computation budget,
and they must not contain interpenetrating or arbitrary teleported cloth.

Provide progress from verified sleeve–limb insertion, terminal completion and
explicit failure/contact costs. Allow purposeful release. The old rule that
the original anchor must remain within 2 cm for the entire episode cannot define
success for a task with deliberate regrasping. Grasp state, unintended garment
drop, insertion loss, final dressing completion and contact risk need distinct
measurements. Ordinary reward/constraint alignment is necessary task definition;
it does not require rerunning the old Stage 0 campaign.

Replay reuses executed experience; it does not create unobserved bimanual data.
Our existing 2,424 branch records lack the actions and successor histories needed
to initialize this as a complete bimanual RL dataset. They remain failure evidence.
Likewise, 16/16 outward continuations preserving the recorded anchor proxy, with
zero valid completions, do not establish that regrasping is necessary or sufficient.
See the [long-continuation record](2026-09-21-feasible-segment-result.md).

No paired full-episode counterfactual rollout is required for every update in
this design. Fewer high-level decisions, shared geometry features and replay
are concrete possible efficiency mechanisms. They do not prove a smaller total
physics budget. Bootstrapping also introduces estimation errors; strong action
search can exploit them. Neither off-policy learning nor a learned model removes
the need to encounter useful transitions.

The simulator is an implementation choice. [Newton's official repository](https://github.com/newton-physics/newton)
includes robot/cloth examples, making it a candidate to consider alongside
existing infrastructure. That verifies feature availability, not batched dressing
throughput, faithful grasp slip, or a training-time advantage over IPC. No engine
migration is selected or performed here. Public rigid-body RL speedups must not
be used as estimates for this cloth task.

An honest cost statement requires

$$
T_{\mathrm{total}}=T_{\mathrm{state\ preparation}}+
T_{\mathrm{physics\ collection}}+T_{\mathrm{updates}}+
T_{\mathrm{evaluation}}+T_{\mathrm{recovery\ overhead}}.
$$

This review provides no measured value for the new task. In particular, it does
not support a promise of hours, two days, or a fixed multiplier relative to the
old task. A workstation limit is a design requirement, not an achieved result.

### Deployment

Deploy the same sensor-conditioned policy and primitive controller. Each cycle
reads cameras/proprioception/available forces, updates history, selects a
point/primitive/parameters, executes with feedback, and observes again. There
is no teacher to distill and no required online IPC search. The trained critic
may remain part of discrete action selection, as in the reference design.

Real deployment additionally needs a calibrated camera/robot/body frame,
observable grasp status and an executable release/regrasp interface. A simulator
spring attachment alone does not establish a real gripper capability. Start any
later hardware work with the corresponding fixture/mannequin task; the current
review does not authorize new trials or claim human safety.

## Why several attractive shortcuts are not selected

- **Copy Bi-Adapt and claim 50 interactions suffice.** Its adaptation budget is
  downstream of pretraining, and its studied objects/actions do not establish
  long-horizon cloth recovery.
- **Use a world model and assume training becomes cheap.** The 2026
  [cloth-unfolding world-model paper](https://arxiv.org/html/2602.16675) uses a
  modified DreamerV2 approach, emphasizes demonstration initialization, and
  reports the limitation of training agents for individual garment types. It is
  useful prior work, not evidence for an inexpensive demonstration-free universal
  dressing policy.
- **Reward more contacts.** Contact coverage does not specify whether cloth
  slides over the body or builds harmful tension. This requires task-relevant
  consequences, not just a remapped contact counter.
- **Predict a sleeve centroid and imitate a geometric teacher.** This reopens
  the explicitly rejected Stage 1 route and leaves support changes unaddressed.
- **Call a manipulation-primitive policy a new algorithm.** Temporal abstraction,
  hybrid actions, downstream values and goal-conditioned replay are established.

The deliverable of this pass is a narrowed task and an explicit, falsifiable
learning hypothesis with a named reference learner. A novel general RL algorithm
has not been established. The strongest next research focus is support-changing
recovery during insertion; further activity remains literature/design work under
the owner's current no-tests instruction.
