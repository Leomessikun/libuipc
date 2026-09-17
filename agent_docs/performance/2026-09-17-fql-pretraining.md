# First dressing FQL pretraining run

The owner authorized training and asked to select the best suitable method from
Sergey Levine's work. This supersedes the earlier hold on new training, but does
not restart abandoned IPC-gradient correction experiments. This is an established
method adapted to the existing dressing interface, not a new algorithm.

## Selection

| Candidate | Evidence and decision for this first run |
|---|---|
| [FQL, Park/Li/Levine, ICML 2025](https://proceedings.mlr.press/v267/park25f.html) | Selected for demonstrated offline-to-online use and one-step actor inference. A behavior flow model regularizes return-based actor learning. The [authors' implementation](https://github.com/seohongpark/fql/blob/master/agents/fql.py) is the objective reference. |
| [RQL, Oberai/Park/Levine, June 2026](https://arxiv.org/abs/2606.17551) | Reports stronger aggregate offline results across 50 OGBench tasks. It trains intermediate flow steps with reversal and expectile value learning; the experiments use action chunking and tune both expectile and behavior regularization. That ranking does not establish the fastest visual dressing learner. A relevant later comparison, not grounds to claim FQL is universally best. |
| [IQL](https://arxiv.org/abs/2110.06169), [Cal-QL](https://arxiv.org/abs/2303.05479) | Established alternatives addressing offline support/value calibration. The earlier local IQL pilot does not settle their performance with these demonstrations. |
| [RLPD](https://arxiv.org/abs/2302.02948) | Relevant prior-data online control; not itself the offline pretraining algorithm requested here. |

Selection is a task/compute judgment, not a measured algorithm ranking. Do not
claim a new contribution or a dressing improvement before complete evaluation.

## Implementation

- `python/uipc_manip/fql.py`: separate behavior flow, one-step noise-conditioned
  actor, dense action-conditioned twin critic and target critic. Ordinary-return
  TD with configurable mean/min aggregation (mean in this run), flow matching,
  detached Euler teacher actions, action-MSE distillation and actor Q gradient.
  No SAC entropy term or IPC derivative. Uses existing point encoders and
  normalized residual trunks; this architecture differs from the paper's MLP/
  image encoders. The name `wang-flow` elsewhere remains unrelated to generative
  flow matching.
- Encodes the behavior observation once per ten-step Euler integration; reuses
  point neighborhoods across forwards. CUDA fused Adam, cached device batches,
  batched scalar logging. Targets, optimizer and RNG are saved in explicitly
  versioned FQL checkpoints; they cannot be loaded as SAC checkpoints.
- `python/uipc_manip/train_fql.py`: body-disjoint offline training and separate
  native evaluation modes. Evaluation initializes Genesis before CUDA matrix
  work and reuses the existing full-episode recorder. The learned flow prior
  is the imitation-only comparison to the one-step RL actor. Both are sampled
  with matched, explicit random seeds; zero latent noise is not called a mean.
- Episode tapes optionally record explicit successors. Reconstruction can retain
  simulator-valid failed source episodes for RL, with BC admission recorded
  separately. Invalid simulator-reset transitions are excluded; real timeout
  successors bootstrap. Neither adjacent episodes nor fabricated zero states
  serve as successors. Group repeated sources on the same split.

## Reward and data

The historical shoulder ray cannot intersect an opening just beyond its origin.
An opt-in `WangRewardConfig.upperarm_extension_m` moves that origin outward;
this run uses 0.05 m, with progress/reward capped at the upper-arm length. Zero
preserves the historical geometry contract. The correction requires a current
opening-triangle intersection and cannot award success from a past maximum.
It is bounded geometric extrapolation, not a proof of garment topology or a
complete reward redesign. The boundary beyond that five-centimetre band, partial-
progress incentives, perception and grasp failures remain limitations.

The first reconstruction process was stopped during preparation and its partial
directory retained. The corrected run replays all 25 existing expert action
sequences once (five garments and bodies 14045–14049), in batches of five. This
is data reconstruction with fresh outcomes, not a repaired teacher or invented
expert labels. Physics/observation/control compatibility is checked before the
explicit reward override. The corrected dataset records its full environment
configuration and true successor observations. Old replay rewards are not mixed
in; the old 945,864-transition replay inventory lacks geometry for exact relabeling.

Training bodies: 14045, 14046, 14047. Validation bodies: 14048, 14049. These are
development splits; prior work has already inspected these bodies. No pristine
research test result or broad regional generalization is claimed. All valid
failed transitions are included, so this run can still be limited by poor data.
New reconstruction outcomes replace source outcomes when computing admission.

## Bounded first protocol

Fresh FQL weights (the SAC checkpoint supplies architecture/environment only),
3,000 updates, batch 128, alpha 100, learning rate 3e-4, gamma .995, tau .005,
ten Euler steps. The dense critic and residual trunks use width 1024 from the
saved reference. Alpha is an initial setting, not a proven optimal coefficient.
Save every 500 updates; select the predeclared final checkpoint for rollout,
not the checkpoint with the best inspected rollout result.

Evaluate actor and learned behavior prior on tshirt_26/14046, tshirt_68/14046,
tshirt_26/14048 and tshirt_68/14049, two full 300-decision rounds with policy
order reversed in the second round. Record final coverage, whole-episode grasp
validity, simulator errors and total time. The first run answers whether value-
based improvement helps the jointly trained prior; it is not a matched SAC/
RLPD comparison or an offline-to-online improvement result. An online collector
integration remains separate work; checkpoint continuation already retains the
FQL learner's objectives and state.

Artifacts: `output/uipc_manip/fql_pretrain_20260917/`.

Validation before learning: 19 focused tests pass (FQL targets, detached flow
integration, updates of all components, exact full-state resume, source split
isolation, invalid reset rejection, timeout successors, reward geometry and
existing expert tests). Python compilation and `git diff --check` pass.
Native reconstruction is running; learning/evaluation results will be appended.
