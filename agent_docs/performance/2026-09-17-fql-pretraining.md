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
validity, simulator errors and total time. The first run compares the complete FQL actor to the jointly trained prior;
it does not isolate the Q term and is not a matched SAC/
RLPD comparison or an offline-to-online improvement result. An online collector
integration remains separate work; checkpoint continuation already retains the
FQL learner's objectives and state.

Artifacts: `output/uipc_manip/fql_pretrain_20260917/`.

Validation before learning: 19 focused tests pass (FQL targets, detached flow
integration, updates of all components, exact full-state resume, source split
isolation, invalid reset rejection, timeout successors, reward geometry and
existing expert tests). Python compilation and `git diff --check` pass.
An additional 39 relevant existing tests pass (four native tests deselected),
with one existing empty-distance warning in a mocked simulator-error test.
The resume equivalence test is bitwise on CPU, not a GPU determinism claim.
Recomputing the corrected metric on 1,806 saved SAC geometry frames preserves
their existing ratios to numerical precision; those frames do not themselves
contain the expert's shoulder overshoot. The synthetic regression exercises
that boundary, out-of-band openings and lateral false positives.

## Preparation and first learner completed

`dataset_shoulder_v1/` contains 25 complete episodes / 7,500 transitions, no
simulator errors, collected in 662.56 s including that process's dataset build
and serialization. Nine episodes pass final coverage and whole-episode grasp
admission: seven of 15 training episodes and two of ten validation episodes.
The loader retains all 4,500 / 3,000 valid train/validation rows, including
failed episodes. Every tshirt_392 reconstruction fails. Original success labels
are not inherited. The earlier aborted setup and implementation/testing time
are outside the 662.56 s preparation measurement.

The first run is in `train_a100_s17/` with log `train_a100_s17.log`.
At 500 updates, elapsed learning-loop time is 55.47 s and validation actor/prior
action MSE is .08124 / .07841 (initial .88963 / .72917). These are fitting
diagnostics, not dressing success or convergence. Sampled GPU utilization is
100%, approximately 22 GiB used and 461–486 W; not an occupancy measurement.

## Completed 3,000-update pilot

Training completes with finite logged diagnostics in 334.00 s including loading,
validation and checkpoint writes (332.03 s learning loop). Final validation
actor/prior action MSE is .06671 / .06416. Peak Torch allocated memory is
12,032,960,512 bytes; device-wide sampled usage includes reserved/runtime memory.

The predeclared final checkpoint was evaluated for two complete rounds, reversing
actor/prior order. Every episode contains 300 decisions and there are no simulator
errors. Evaluation including world setup costs 459.34 s. Successful reconstruction,
training and this evaluation total 1,455.90 s (24.26 min), excluding the aborted
initial setup and implementation/testing work already identified above.

| Policy / split | Episodes | Final coverage >= .7 and whole-episode grasp <= 2 cm | Same criterion with coverage retained for final 20 decisions | Mean final coverage | Whole-episode valid grasp | Historical paper filter |
|---|---:|---:|---:|---:|---:|---:|
| FQL actor / all | 8 | 4 | 4 | .47719 | 8 | 0 |
| FQL actor / training body | 4 | 2 | 2 | .50000 | 4 | 0 |
| FQL actor / withheld bodies | 4 | 2 | 2 | .45439 | 4 | 0 |
| Flow behavior prior / all | 8 | 0 | 0 | .20427 | 2 | 0 |
| Flow behavior prior / training body | 4 | 0 | 0 | .33213 | 2 | 0 |
| Flow behavior prior / withheld bodies | 4 | 0 | 0 | .07640 | 0 | 0 |

FQL succeeds on tshirt_26/14046 in both rounds (coverage 1.0/1.0) and withheld
tshirt_68/14049 (1.0/.81755). It fails on tshirt_68/14046 and withheld
tshirt_26/14048 in both rounds (zero final upper-arm coverage). These are four
configurations repeated twice, not eight independent configurations. All FQL
episodes trigger `early_turn`, including its geometric successes; none passes
the stricter historical paper filter. The prior reaches .873 transiently on
tshirt_26/14048 in one round but finishes at zero with an invalid grasp. Peak
coverage would misrepresent that result.

This is preliminary evidence for the complete FQL actor relative to its learned
flow behavior prior under the declared corrected geometric metric. It does not
isolate the Q term: one-step distillation versus iterative action generation is
also different. A one-step distillation-only control would be needed for that
causal claim. There is no matched SAC/RLPD comparison, multiple training-seed
result, pristine test-set result, real-robot result, or robust dressing claim.
Do not compare this corrected-reward/data protocol directly to historical SAC
success counts. Three thousand updates do not establish value convergence.

Artifacts: `summary.json`, `train_a100_s17/result.json`,
`train_a100_s17/final.pt`, and `eval_a100_s17/result.json` under the root above.
The evaluation completion flag was added to the CLI; this initial run's result
was marked complete only after its process exited successfully and all four
evaluation groups / 16 complete episodes were verified.

## Completed 30,000-update continuation — 2026-09-18

The full-state continuation and automatic evaluation both finished. No FQL
training process remains. The learner performed 27,000 additional updates on
exactly the same data, reaching 30,000 total gradient updates; these are not
new physics transitions. Continuation took 2,938.85 s and evaluation 381.88 s.
Both training phases together took 54.55 min. Successful preparation, both
training phases and both evaluations together cost 79.61 min, excluding the
previously identified aborted setup and engineering/testing work.

All four evaluation groups completed, with 16 full 300-decision episodes and
no simulator errors. Artifacts: `continue_a100_s17_30k/result.json`, its
`final.pt`, `eval_a100_s17_30k/result.json`, and `summary_30k.json`.

| Policy / split | Episodes | Final coverage >= .7 and whole-episode grasp <= 2 cm | Mean final coverage | Whole-episode valid grasp | Historical paper filter | Paper filter AND valid grasp |
|---|---:|---:|---:|---:|---:|---:|
| FQL actor / all | 8 | 5 | .63059 | 7 | 0 | 0 |
| FQL actor / training body | 4 | 4 | .94996 | 4 | 0 | 0 |
| FQL actor / withheld bodies | 4 | 1 | .31123 | 3 | 0 | 0 |
| Flow behavior prior / all | 8 | 1 | .31746 | 3 | 1 | 0 |
| Flow behavior prior / training body | 4 | 1 | .52743 | 2 | 1 | 0 |
| Flow behavior prior / withheld bodies | 4 | 0 | .10749 | 1 | 0 | 0 |

The historical paper filter alone does not enforce grasp validity: its single
prior pass has 3.08 cm maximum tracking error and fails the combined criterion.
Every FQL episode triggers early_turn. Thus geometric success is not yet a
validated robust-dressing result. The tshirt_26/14048 actor never reaches any
upper-arm coverage in either round (maximum forearm coverage .852/.910).
The tshirt_68/14049 actor fails with 2.74 cm tracking error in one round and
succeeds in the other. These are distinct observed failure patterns; an elbow-
only cause, value extrapolation, and insufficient observation history remain
hypotheses, not diagnoses.

Compared with 3,000 updates, training-body successes rise 2/4 to 4/4 while
withheld-body successes fall 2/4 to 1/4. Four cells, two evaluation rounds and
one training seed cannot establish a significant improvement or overfitting.
Longer offline optimization has not established improved generalization.

## Research plan after the completed pilot

Objective: one shared dressing policy that uses existing IPC experience to
reduce fresh simulation and total single-workstation time to robust success.
This is currently established FQL pretraining, not a novel RL algorithm and
not an implemented offline-to-online training loop.

1. Audit success and the repeated failures before another long run. Inspect
   sleeve/arm geometry, early_turn timing, commanded movement and grasp tracking
   for the failed withheld configurations and reported successes. Check whether
   the early-turn flag identifies an actual bad dressing route. Do not remove
   that criterion just to increase success. Freeze the reward/evaluation contract
   for subsequent matched comparisons; no retrospective metric switching.
2. Isolate the learning benefit. Add a one-step distillation-only control with
   the same data, architecture, latent sampling and update budget as FQL, disabling
   only the actor's Q objective. Retain the behavior-flow comparison but do not
   attribute its difference from FQL solely to Q. Repeat the compact protocol
   across training seeds before expanding the study. Keep withheld bodies out of
   gradient updates and choose new untouched configurations for final testing.
3. Test whether pretraining saves subsequent RL cost. Connect the existing native
   collector to the same FQL learner, preserving targets/optimizer and genuine
   terminal/timeout semantics. Keep prior and fresh training experience separately
   accounted for. Compare pretrained FQL continuation with FQL from scratch and
   SAC under the same simulator, reward, observations and task distribution;
   add the prior-data SAC/RLPD control to separate prior-data use from the choice
   of algorithm. Report success versus fresh physics transitions AND total wall
   time, including preparation/pretraining/evaluation. Declare bounded budgets
   before launching; do not treat offline checkpoint resume as online learning.
4. Design a new mechanism only around a reproducible baseline limitation. The
   research question is how to turn sparse successful contact experience into
   reliable recovery on configurations absent from training, using little fresh
   expensive simulation. If the audit finds missing recovery transitions, first
   test targeted recovery data against equal-cost generic collection. If data is
   adequate but value-based actor improvement fails, test that specific learning
   failure instead. Neither explanation is established yet. FQL plus IPC alone
   is not an algorithmic contribution. Broader contact-task benchmarks and real-
   robot transfer follow a repeatable dressing gain, with a prior-art check
   before any novelty claim.

Immediate next work is the failure/metric audit and matched one-step control,
not an indefinite extension beyond 30,000 updates or a solver-gradient redesign.
No new training was launched while preparing this research-plan update.
