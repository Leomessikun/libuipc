# Research decision after the motion-pretraining pilot

Status: research assessment on 2026-09-18. No learner, physics experiment,
reward change or new algorithm was launched by this review. The recommendation
below is a proposed gate, not an executed experiment or a claim of improvement.

## Decision

Stop scaling the current cloth-motion encoder initialization recipe. Continue
the broader objective of a shared dressing policy trained efficiently from
existing experience and limited fresh IPC interaction, subject to a bounded
diagnostic gate. There is no evidence yet for a novel successful RL algorithm,
a robust dressing policy, or an IPC advantage over another simulator at equal
cost. Those are separate research claims.

The immediate research question is whether additional experience can teach the
policy useful decisions at its actual contact failures. Increasing the number
of gradient updates on unchanged data does not answer that question. The
completed FQL continuation is offline; online FQL collection remains unimplemented.

## What the local evidence establishes

The [motion pilot](2026-09-18-motion-pretraining.md) reused five training
episodes with material geometry, producing 1,480 overlapping windows. It
transferred only the actor encoder into an otherwise fresh FQL agent. The motion
decoder was discarded; it did not guide action selection, train the critic, or
remain as an auxiliary loss during RL.

| Initialization | Coverage plus whole-episode grasp passes | Mean final coverage |
|---|---:|---:|
| Random FQL | 2/8 | .294 |
| Geometry encoder | 1/8 | .409 |
| Motion encoder | 0/8 | .071 |

All variants received 3,000 gradient updates. There was one training seed and
four configurations, each repeated twice. The motion policy never reached
peak coverage .7. This is sufficient reason not to scale this recipe; it is
not a statistically established ranking of pretraining methods. The historical
random baseline scored 4/8 and its repeated training was not bitwise identical.

The held-out motion error was 11.74 mm, against 15.95 mm for zero motion and
12.39 mm for constant velocity. The latter has an extra previous material
frame, so this is a useful stronger diagnostic rather than an identical-input
comparison. Near the elbow, the learned error was 5.20 mm versus 2.45 mm for
zero motion. Exact sleeve-opening queries numbered only 32 in training and
nine in validation. Global motion fitting did not establish accurate behavior
at the critical region, much less correct ranking of recovery actions.

Three task/data issues precede another algorithm claim:

1. `dressing_env.py` evaluates grasp tracking against 2 cm, while
   `dressing_reward.py::wang_progress` has no direct grasp-tracking term.
   Twelve of the 24 pilot rollouts violate whole-episode grasp validity.
   This is an objective/evaluation mismatch, not proof it caused every failure.
2. The historical early-turn predicate is already true at reset for both
   withheld evaluation configurations. The bounded shoulder extension also
   leaves possible ray-intersection discontinuities. Validate physical
   completion against saved geometry before attributing every flag or lost
   coverage reading to the policy. Preserve the old results and version any
   subsequent contract change; do not silently relabel old rewards.
3. There is substantial data on disk, but its usefulness differs by objective.
   The [inventory](2026-09-17-prior-data-inventory.md) found 945,864 historical
   transitions, predominantly failures and without the geometry needed for
   current reward relabeling. The current FQL dataset has 25 reconstructed
   episodes, 15 for training and 10 for validation. Only five training
   trajectories supplied the material-motion pilot. These are three different
   datasets, not evidence that the owner failed to collect trajectories.

The [earlier review](2026-09-18-research-plan-review.md) identifies long stalls
and teacher failures on withheld bodies. Its causal wording needs qualification:
teacher difficulty confounds the learner's generalization gap; it does not prove
the gap is mostly caused by the teacher. One recorded action at a stalled state
does not mathematically prevent critic generalization to alternatives, but it
does not supply a measured comparison of their outcomes either.

## What the literature changes about the decision

- **PointZero** reports a benefit from large-scale track pretraining: its
  same-architecture comparison with downstream track supervision improves
  average success from 80.5% to 88.2% on three manipulation tasks. It uses 2.9M
  synthetic frames and about two days on eight H100s per variant. Its policy
  transfer is imitation learning, not dressing RL. Our small actor-only test
  neither reproduces nor refutes that result. Its public code repository still
  says release is forthcoming as checked today.
  [Paper](https://arxiv.org/html/2609.19142v1),
  [repository](https://github.com/Duisterhof/pointzero).
- **3PoinTr** uses predicted point tracks as an explicit input to its action
  policy. That is a stronger connection between prediction and control than
  discarding our decoder after initialization. It is prior art for such a
  connection, not evidence that another integration will fix dressing.
  [Paper](https://arxiv.org/html/2603.08485v2).
- **Is Value Learning Really the Main Bottleneck in Offline RL?** separates
  value learning, policy extraction, and generalization to states visited by
  the learned policy. Its results support measuring the latter directly instead
  of assuming a critic defect from poor rollouts. The paper does not identify
  our bottleneck on our behalf.
  [Paper](https://arxiv.org/html/2406.09329v2).
- **The Value Equivalence Principle** formalizes why a model useful for control
  need not reconstruct all state transitions accurately. Better global cloth
  displacement error is therefore an insufficient acceptance criterion for a
  control-oriented model. Task-relevant model learning itself is established
  prior art.
  [Paper](https://arxiv.org/abs/2011.03506).
- **Data Scaling Laws in Imitation Learning for Robotic Manipulation** finds
  strong benefits from distinct objects/environments compared with repeated
  demonstrations of the same configurations. This supports coverage audits;
  its numerical thresholds cannot be transplanted to dressing.
  [Paper](https://arxiv.org/abs/2410.18647).
- **RFCL** already uses demonstration-state resets and reverse/forward
  curricula to improve sample efficiency. Selecting difficult states or adding
  recovery examples is not sufficient novelty. Its rigid manipulation results
  do not establish that our cloth restore is reliable or economical.
  [Paper](https://arxiv.org/html/2405.03379v1).
- **Dressing in Motion**, a September 2026 preprint, reports a dressing policy
  trained from static demonstrations with motion-aware trajectory adaptation,
  including a nine-participant study. This is evidence of continued progress
  in the application, and another relevant comparison for a future deployment
  claim. It does not validate our learning pipeline or isolate elbow recovery.
  [Paper](https://arxiv.org/html/2609.04759v1).

## One bounded gate before more method development

First validate the task contract using existing geometry and traces: physical
arm-through-sleeve completion, sustained progress, and retained grasp. Report
teacher outcome separately from learner outcome. Establish which observed
failures are physical stalls, grasp failures, or measurement failures. Any
reward or controller change must apply equally to later control and treatment
arms and have a new data contract.

Then budget at most one hour of additional native simulation for an initial
decision diagnostic, including setup, prefix replay and restoration. Select a
small fixed set of distinct training/development failure states before seeing
the outcomes. Compare the current continuation with a few finite alternative
routes at matched movement and continuation budgets. Repeat each continuation;
retain full transitions and geometry. If restoration changes outcomes too much
to resolve an action preference, stop this test and localize restore versus
fresh-prefix variability. Numerical spread is not proof of physical chaos or
of an impossible RL task.

The existing recovery experiment already found one useful route and its BC
transfer did not establish improvement (3/8 versus 2/8). Therefore another
single recovered trajectory is not a pass. The gate asks whether useful
preferences recur across configurations and can be inferred from deployable
observations. An inexpensive held-out prediction probe may test this last
condition; a probe win alone is not policy improvement. Full simulator state
can provide diagnostic labels but cannot become an undeclared deployment input.

Only if that gate passes should we connect online FQL and test whether ordinary
learning on real collected transitions converts the information into improved
completion. Keep the learner fixed initially. Compare pretrained and fresh FQL
under a common total workstation budget; compare selective collection against
ordinary collection before attributing a gain to selection. A subsequent
SAC-with-prior-data baseline is necessary before claiming a better RL algorithm.
Use multiple training seeds and new body/garment configurations, report paired
outcomes, and count preparation, restoration, training and evaluation time.
Declare the exact rollout count and runtime cap before launching that comparison.

Stop dynamics-guided method development if the diagnostic cannot resolve useful
decisions within budget, or if the subsequent matched learner comparison yields
no repeatable benefit. Retain the simulator, dataset and baseline infrastructure.
Do not automatically launch a larger model, more epochs, more bodies, or a sweep
of algorithms in response to a negative result.

## Practical and scientific limits

IPC supplies forward transition samples and privileged diagnostic geometry.
Its solver fidelity does not create successful exploration, identify actions
absent from data, or calibrate simulated cloth to a real garment. A future
IPC-specific advantage needs equal-cost comparison with another solver and
physical validation. Deployment also needs compatible observations and robot
actuation; simulator success is not end-to-end hardware validation.

The potential contribution is a demonstrable reduction in total cost to learn
contact-failure recovery across tasks. Encoder pretraining, an auxiliary loss,
or reset curricula alone are not that contribution. Confidence is high that
the present motion-initialization recipe should not be scaled; evidence for
the broader efficient-learning hypothesis remains incomplete.

During this review, separate `measure_predictability_horizon.py` diagnostics
were observed running for tshirt_26/14046 and tshirt_392/14046. They were not
started or modified by this review, and their incomplete outputs are not used
as results here. They may inform the repeatability gate when complete; they
do not themselves train or evaluate an improved policy.
