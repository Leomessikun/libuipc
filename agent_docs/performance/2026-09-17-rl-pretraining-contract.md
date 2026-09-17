# RL pretraining infrastructure: correction after reviewing CS 285

Implementation update: the owner subsequently authorized training. The
[FQL implementation and measured pilot](2026-09-17-fql-pretraining.md) now exist,
with offline continuation to 30,000 updates running. The review/proposal below
records the earlier reasoning; full online collector integration is still open.

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

## Concrete first training design: shared FQL pretraining and continuation

Proposed after the owner's follow-up asking how to train the policy. This selects
an established learner for a concrete experiment; it is not implemented, a new
algorithm, or evidence of better dressing. The preceding review's statement that
no method was selected describes that earlier review. Keep the withdrawn
sleeve-goal architecture and abandoned IPC correction runs stopped.

**Learn one policy across configurations from existing compatible experience,
then improve it with fresh IPC interaction.** Use FQL as the first explicit
pretrain-to-online learner. It has a behavior flow model, return critics, and a
separate one-step actor. The actor maximizes predicted return while remaining
close to the flow model's sampled actions. Continue these objectives online;
do not export an ordinary-return critic and silently resume SAC's soft-return
updates. Only the one-step actor is needed for execution. This is the published
[FQL structure](https://arxiv.org/html/2502.02538v1), not our contribution.

### Data and reward contract

- Start with the existing point-cloud/proprioception input and six-dimensional
  gripper command/controller contract. Keep history length fixed for the first
  comparison. Joint training already exists; eliminating regional teachers is
  a training choice, not architectural novelty.
- Correct and validate the shoulder-overshoot reward/success issue using saved
  geometry before producing a new reward version. Physical completion and
  retained valid grasp must determine success; taking maximum progress alone
  is not a fix. Give every comparison the same objective and evaluator.
- Build a manifest of prior data by observation, action, physics, reward version,
  episode/configuration identity and available successors. This is adaptation
  of existing files, not a request to collect a new corpus. Cross-solver data
  require an explicit compatibility/transfer decision, even for action labels.
- Use successful and valid failed transitions for value learning when their
  rewards and successors satisfy that contract. Do not import distillation's
  success-only admission rule into the transition loader. Keep prior and newly
  collected transitions separately identifiable; log their actual sampling mix.
- Old reward values cannot be mixed with corrected values. Relabel only when
  the recorded fields determine the corrected reward exactly. Otherwise,
  compatible observation/action pairs can still support the behavior prior,
  but those rows are not corrected-reward Bellman examples. A changed success
  termination rule also changes masks/boundaries, not only reward scalars.
- The inspected reconstructed native episode has 300 observations, actions,
  rewards and privileged vectors, with no extra final observation. Adjacent
  pre-action observations can supply internal successors. Do not fabricate the
  final successor, join episodes, or treat a timeout as a terminal failure;
  exclude an incomplete bootstrap row from Q updates. Existing SAC replay has
  its own explicit successor/boundary contract. It lacks privileged geometry,
  which prevents assuming that all its rewards can be geometrically relabeled.
- Split by source episode and body/configuration, grouping repeated replays.
  Already inspected development bodies are not untouched final test bodies.

### Learning, integration and compute

1. Cache admitted existing data and train the behavior prior, critics and actor
   with the FQL objectives. This is reward-based pretraining, not just the
   earlier action-MSE fit. Reuse current observation processing, normalization
   and configured dense-Q/residual modules where compatible; document any
   departure from the reference learner.
2. Integrate this learner with the existing collector and replay boundary
   handling. Continue the same objectives with mixed prior/fresh IPC experience,
   preserving targets and optimizer state in an explicit FQL checkpoint. SAC
   remains a separate baseline; this is not a SAC option that adds IPC gradients.
3. IPC supplies real simulator transitions and complete-episode evaluations.
   Network updates use cached batches without an IPC solve or derivative per
   update. Batch/preload to the single GPU where memory permits; tune update
   reuse against held-out rollout gain and wall time. Flow-model training and
   its action sampling also cost compute. High utilization alone is not success.
4. Deploy the actor through the existing command/controller interface. Neither
   flow integration, a critic nor IPC is required for actor inference. This
   does not establish real-world transfer: the real observation pipeline and
   controller must meet the training contract and be validated separately.

The reason to test this is specific: the audited ordinary SAC did not consume
the expert folder, successful exploration was scarce, and environment plus
evaluation time dominated its run. These facts justify testing more useful
learning per IPC interaction. They do not prove that FQL fixes perception,
partial observability, elbow recovery or critic error. The small BC result also
does not demonstrate that action multimodality was its limiting factor.

### Decision and contribution

The first complete comparison is this pretrain-plus-continuation recipe against
ordinary SAC and [RLPD](https://arxiv.org/html/2302.02948v3), which tests whether
online reuse of prior data suffices without offline initialization. Match task,
reward, sensors and configuration splits; give FQL/RLPD the same admissible prior
data. Report prior-data preparation cost, new IPC decisions, all learning and
evaluation time, and repeated full-episode success/grasp validity. Compare the
pretraining ablation at equal total cost, not equal online steps alone. Existing
unmatched BC/SAC numbers are not this experiment's result. First obtain a bounded
complete curve; expand seeds/configurations only if it merits that cost.

This is an implementable pretraining structure, with no demonstrated algorithmic
novelty yet. Action chunking is already addressed by
[Q-chunking](https://arxiv.org/abs/2507.07969), and contact-dependent adaptive
chunk duration has close prior art in
[Adaptive Q-Chunking](https://arxiv.org/abs/2605.05544). Do not relabel those as our
new algorithm. A contribution would need a measured remaining failure and a
specific improvement beyond these references at matched workstation cost.
No learner code, native simulation or training was run for this design update.
