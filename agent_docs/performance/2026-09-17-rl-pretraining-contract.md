# RL pretraining infrastructure: correction after reviewing CS 285

Status: literature and source review, not implementation or a new training result.
The owner challenged the sleeve-transfer recommendation and referred to
[Sergey Levine's Berkeley course](https://rail.eecs.berkeley.edu/deeprlcourse/).
The preceding recommendation to make a new garment-goal architecture the default
next experiment is withdrawn. The goal remains fast, easy training of a robust
dressing policy on one workstation, without the cost of independent regional
teachers. A joint policy path already exists in this repository.

## Corrected understanding

Pretraining is a stage and a transfer contract, not a particular loss. It can
learn a representation, a behavior prior, a value function, a dynamics model,
or combinations of them. It can include simulation and RL itself. Offline RL
is one option, not the definition of pretraining. BC can also provide useful
initialization for subsequent RL; standalone BC success does not measure how
much downstream learning it saves. Low prediction loss is likewise insufficient.

The project should specify what reusable knowledge prior experience supplies,
how reward-based policy improvement consumes it, and whether that reduces total
time to robust dressing. The normal SAC audit does not justify abandoning this
objective or requiring a new action space before testing it.

## Lessons from the primary sources

| Material inspected | Consequence for our design |
|---|---|
| [Lecture 17](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-17.pdf), slides 6–11, 17–20 | Offline RL can improve on recorded behavior, including through compatible behavior fragments, but must handle actions outside dataset support. The lecture explicitly distinguishes offline RL from pretraining more generally. |
| [Lecture 18](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-18.pdf), slides 23–30 | Offline-to-online value recalibration can lose performance. Prior-data online learning and behavior-prior/value-based learning are different viable designs. |
| [Lecture 16](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-16.pdf), slides 17–25 | Policy learning with a model need not differentiate through dynamics. Short model rollouts can supply transitions for value learning; model error and sampling distribution matter. |
| [Lecture 25](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-25.pdf), slides 35–36 | Diverse prior experience can train reusable models, skills or goal-conditioned behavior before downstream adaptation. A successful optimized trajectory alone does not demonstrate this learning benefit. |
| [RLPD](https://arxiv.org/html/2302.02948v3), sections 4–5 | SAC-based online RL can use prior data through separate buffers, balanced sampling, normalized critics and ensembles for increased reuse. It does not require offline optimization first. Prefilling ordinary SAC replay is not the complete method. |
| [FQL](https://arxiv.org/html/2502.02538v1), section 3 | A flow behavior model regularizes a separate one-step actor trained to maximize Q. This combines data fitting with RL improvement, without iterative action generation at deployment. Its actor gradient is not an IPC derivative. |
| [Cal-QL](https://arxiv.org/html/2303.05479v3), sections 4–5 | Conservative values may be at an unsuitable scale for online improvement. Calibration addresses that problem; unsuccessful IPC gradient labels do not refute ordinary critic learning. |

The [course's offline-to-online project](https://rail.eecs.berkeley.edu/deeprlcourse/static/misc/offline_to_online_rl_default_final_project.pdf)
also separates initialization, replay mixing, fresh collection and online
improvement. Its benchmark budgets are not IPC dressing budgets. These sources
establish methods and failure mechanisms elsewhere, not a dressing result here.

## Actual repository contracts

| Existing path | What learns/transfers | What the result does not establish |
|---|---|---|
| [`pretrain_offline.py`](../../python/uipc_manip/pretrain_offline.py), [`SACAgent.initialize_representation`](../../python/uipc_manip/sac.py) | Successor/reward prediction trains the actor encoder and optional history. Initialization transfers those modules only. | The action head, critic, temperature and optimizers remain fresh by design. This is valid representation pretraining, not an already trained actor-critic. |
| [`distill.py`](../../python/uipc_manip/distill.py) | Supervised actions train the actor. | Critic explicitly untrained. The small measured native run uses one training body; its BC rollout result does not measure later RL acceleration. |
| [`offline_rl.py`](../../python/uipc_manip/offline_rl.py) | IQL trains Q/V and an advantage-weighted actor; BC/replay-SAC are controls. | The completed 2,000-update test uses the previous SAC run's own 125,016-transition replay, without expert trajectories or integrated online IQL continuation. It is not the intended demonstration-data pretrain-to-RL experiment. |
| [`pretrain_wang.py`](../../python/uipc_manip/pretrain_wang.py) | Regional teachers/student or a joint policy; online SAC, optional representation initialization and recovery actor supervision. | Audited ordinary SAC did not consume the expert folder. Recovery action MSE is not a prior transition buffer training the critic. |

These are useful components. There is no universal requirement to pretrain
every network. The missing evidence is a controlled demonstration that a
declared initialization/data-reuse recipe reduces downstream learning cost.
The current actor name `wang-flow` does not denote FQL's generative flow model.

The [SAC audit](2026-09-17-normal-sac-rollout-audit.md) documents stalls and loss
of useful sleeve alignment, including grasp-valid failures, but does not isolate
critic error as their cause. The IQL control's 0/4 result remains negative
evidence for that specific warm-start/replay setting, not a verdict on all RL
pretraining. Do not rerun it unchanged and call it a new approach.

## Common infrastructure to establish before another algorithm proposal

1. **Task and reward.** The newer [sleeve/reward audit](2026-09-17-sleeve-path-audit.md)
   records a shoulder-overshoot discontinuity. Specify completion and retention
   independently of this ray-intersection failure before comparing learning.
   Validate any correction against saved geometry; retain the original metric
   for comparison. Peak coverage alone cannot certify retained, valid dressing.
   This metric issue is not proof of the main cause of SAC's plateaus.
2. **Experience.** Use compatible observations, actions, rewards, true successors,
   episode boundaries, configuration identity and physics provenance. Valid
   failed transitions can teach dynamics/value without being expert action
   labels. Behavior fragments can be combined only with sufficient compatible
   state coverage; visually similar cloth is not necessarily the same physical
   state. Different solvers do not silently share a Bellman operator.
3. **Initialization.** Declare exactly which modules and objective transfer,
   including targets, optimizers and entropy settings. Ordinary and soft returns
   differ; switching IQL to SAC is not a transparent checkpoint resume. A good
   actor can also begin with a fresh critic, but that choice must be evaluated.
4. **Improvement.** Specify how prior data and fresh IPC experience contribute
   to reward-based updates. Offline optimization needs distribution-shift
   handling; RLPD instead learns online with prior data. Neither requires
   differentiating IPC. A learned surrogate would be a separate model-based
   extension with its own validation, not a prerequisite.
5. **Compute and evaluation.** Cache data and schedule actor/critic updates
   separately from collection. More updates can reduce simulation needs but
   also overfit or increase runtime. Compare complete downstream learning
   curves, held-out dressing/grasp success, new IPC decisions and total wall
   time, including preparation and evaluation, on the single workstation.

First inspect existing files for those transition fields and reuse current
loaders/replay implementations. This is not a request for a new corpus. In
particular, the main SAC replay has no privileged geometry, but its observed
transitions can still support model-free RL; a proposed full-geometry objective
would impose an additional data requirement. Imitation performance is not
universally bounded by the teacher's aggregate success rate, and that rate is
especially not an upper bound on subsequent reward-based improvement.

The reference comparison should establish whether compatible prior experience
improves RL before claiming a new algorithm. RLPD is relevant when new IPC
interaction is available; FQL supplies a reference for explicit offline
initialization followed by online improvement. These are alternatives to compare,
not instructions to implement both now or evidence that either solves dressing.
Keep observation/action/controller semantics fixed when testing data reuse.
The potential contribution must address a measured limitation beyond these
references at matched total cost. No new loss, world model or action space is
selected by this review, and sim-to-real transfer remains unproven.

The review began at `6719f443`; documentation commits through `2a41a2fd` arrived
in the shared workspace during the review and were preserved. Read the cited
lecture PDFs, selected slide diagrams, primary papers and local source/contracts.
No training, native evaluation, algorithm changes or new success numbers.
Abandoned IPC correction experiments remain stopped.
