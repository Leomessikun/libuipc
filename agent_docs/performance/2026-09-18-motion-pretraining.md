# Bounded cloth-motion pretraining experiment

The owner approved the PointZero-inspired experiment on 2026-09-18. The question
is whether an encoder trained to predict physical cloth motion makes the same
FQL learner a better dressing policy. This is an adaptation of a pretraining
idea, not a reproduction of [PointZero](https://arxiv.org/abs/2609.19142), a new
RL algorithm, or an IPC-gradient correction.

## Existing geometry is sufficient for this pilot

The usual FQL episode tapes have no persistent material identities. However,
the prior SAC rollout audit saved aligned observations, commands, tool anchors
and all material vertices at 301 frames for six complete 300-decision episodes:

- `output/uipc_manip/policy_results_audit_20260917/sac_geometry/`: tshirt_26/14046,
  tshirt_68/14046, tshirt_26/14047, tshirt_68/14048.
- `output/uipc_manip/policy_results_audit_20260917/sac_training_cells/`:
  tshirt_26/14038, tshirt_68/14018.

No new simulation is required to prepare tracks. Bodies 14048/14049 remain
excluded from gradient updates; 14049 has no saved motion episode. Five episodes
give 1,480 training windows; tshirt_68/14048 gives 296 development windows.
Overlapping windows are not independent demonstrations. These are previously
inspected development configurations, not untouched research test data.
The data covers two garments and does not establish broad cloth dynamics.

The source physics, observations and action/controller configuration must match
the FQL dataset, apart from run/cell identifiers and reward. Old source rewards
are unused; no incompatible rewards enter Bellman training. Source episode
steps, reset flags, finite values, anchors and geometry lengths are validated.
Source files and the derived manifest carry SHA-256 identities.

## Representation and correspondence

`python/uipc_manip/motion_pretrain.py` builds five-decision (0.5 s) windows.
Each start frame matches visible cloth voxel centroids to nearest material
vertices within 3 cm, deduplicates, then samples up to 32 IDs without inspecting
the future. Labels follow those exact vertex IDs through five successors.
The decoder's query coordinates are exact starting material positions relative
to the starting tool; it predicts world-axis displacements, so later tool motion
cannot be mistaken for material motion. This is near-visible material sampling,
not a claim that independently voxelized observations provide exact tracks.
Mean observation-to-vertex distance is about 5 mm. Query positions are privileged
training-only inputs. The encoder always receives the original partial observation.

The existing FQL actor's segmentation encoder and tool-point readout are reused.
A disposable MLP decoder receives that latent, the recorded command sequence,
query positions and learned query slots. Motion uses displacement MSE scaled by
5 cm. The geometry control receives zero query coordinates and zero commands;
it reconstructs the current unordered query cloud using symmetric Chamfer loss,
with coordinates scaled by 0.5 m. It cannot copy target coordinates. Its output
heads/slots have the same shape, but its objective is intentionally different.
Padded targets do not contribute. Geometry and motion loss numbers are not
directly comparable.

Only **actor.encoder** transfers. The behavior prior, actor trunk, dense critic,
target critic and RL optimizer start identically from the declared RL seed.
The critic encoder has a different action-conditioned architecture; it is not
silently replaced or partially mapped. The decoder, future commands and material
geometry are discarded at deployment. This first experiment tests actor-encoder
initialization, not critic pretraining, imagined experience, or an auxiliary loss
continued during RL. Those extensions require evidence from this bounded test.

## Predeclared comparison

Three variants, seed 17:

1. Random actor encoder (ordinary FQL).
2. Geometry encoder pretraining: 1,500 updates, batch 128, Adam 3e-4.
3. Motion encoder pretraining: identical pretraining counts and learning rate.

Every variant then receives 3,000 ordinary FQL updates, batch 128, alpha 100,
with the same architecture, random seed and original `dataset_shoulder_v1`
(4,500 RL training rows, 3,000 body-disjoint validation rows). The reward retains
the existing declared 5 cm shoulder extension. Select final checkpoints by update
count, never by inspected rollout outcomes. No validation samples enter updates.

Evaluate each actor for two full 300-decision rounds on the same four cells as
the previous FQL pilot: tshirt_26/14046, tshirt_68/14046, tshirt_26/14048,
tshirt_68/14049. Report coverage plus whole-episode grasp <=2 cm, persistent final
coverage, and the historical early-turn filter separately. Its geometry/route
limitations remain; do not call eight repeated-cell episodes a robustness study.
No extra behavior-prior evaluation is needed for this actor-initialization test.

Motion diagnostics include non-oracle held-out track error, zero-motion error
and a shuffled-command diagnostic. Better track fitting alone is not success:
the transferred policy must improve dressing or learning cost. Count pretraining
and evaluation time, derived-data preparation, and disclose the sunk costs of
the reused datasets (240.69 s geometry collection, 662.56 s FQL reconstruction).
The pretraining variants receive extra computation; report that overhead rather
than claiming an equal-wall-time win from equal RL update counts.

## Implementation and checks

- `motion_pretrain build` creates material windows; `pretrain` supports motion
  and geometry objectives and emits an explicit encoder-only transfer checkpoint.
- `train_fql train --encoder-init` imports only the actor encoder and validates
  architecture, environment and held-out body provenance. It is mutually exclusive
  with full-state resume. A resumed FQL checkpoint retains its transfer provenance.
- Fourteen focused tests pass: the existing seven FQL tests plus correspondence,
  terminal-frame use, future-input isolation, geometry shortcut prevention,
  padded-target masking, encoder gradient/transfer isolation and ordinary FQL
  checkpoint inference after transfer.
- CUDA smoke tests complete two prediction updates and two FQL updates after
  transfer, with finite diagnostics. They are engineering checks, not results.

Artifacts: `output/uipc_manip/motion_pretrain_20260918/`.
Track preparation completes in 1.81 s without physics rollout. The bounded
three-variant experiment is complete; results and limitations follow below.

## Completed representation pretraining and diagnostic audit

Both pretraining variants finish their predeclared 1,500 updates: geometry
37.32 s, motion 36.99 s (including initial/final prediction checks and writes).
Peak Torch allocation is 3.35 GB. The geometry reconstruction validation loss
falls from .54661 to .01936; this is a fitting diagnostic, not policy quality.
The motion model's held-out track MDE is 11.74 mm, versus 15.95 mm for zero
motion and 12.39 mm for a constant-velocity extrapolation from the previous
material frame. Thus the gain over the stronger simple predictor is only about
5.3%, compared with 26.4% over zero motion. Constant velocity gets a previous
frame that the single-frame model does not receive. First-frame velocity is zero.

A post-training audit uses the final checkpoint on CPU, with no parameter updates
or checkpoint selection. Shifting command sequences by half the held-out dataset
raises MDE to 23.39 mm; zeroing learned point features while retaining the known
observation extras raises it to 14.22 mm. These are input-reliance diagnostics,
not validation against counterfactual physical trajectories. The training-time
within-batch one-row shuffle was weak because adjacent command windows overlap.

The whole-cloth mean hides a regional limitation. For 1,024 held-out point/window
pairs starting within 10 cm of the elbow, learned error is 5.20 mm versus 2.45 mm
for zero motion. This subset is largely stationary; the model has not established
accurate near-elbow dynamics. Proximity is a geometric diagnostic, not a contact
label. Only nine held-out and 32 training point/window pairs correspond exactly
to sleeve-opening boundary IDs, so no confident opening-specific conclusion is
possible. Overlapping windows further limit independence. These observations
must accompany any aggregate prediction improvement.

Artifacts: `prediction_audit.json`, `physical_prediction_baselines.json`, and
both `pretrain_*_s17/result.json` under the experiment root. These additional
post-hoc diagnostics do not change the fixed training/evaluation protocol.

## Historical early-turn metric caveat

Recomputing the early-turn heuristic from the saved initial privileged state of
the prior 30k FQL evaluation shows both withheld configurations already satisfy
it at reset, before the policy acts. Their recorded first decisions also flag it.
The two training-body configurations start outside the flagged region and first
trigger it during policy execution. The heuristic tests the tool's route, not
cloth topology. Therefore a zero strict pass count remains reportable but cannot
alone show that the policy caused an invalid elbow route in the withheld cases.
No metric was changed for this comparison. See `evaluation_metric_audit.json`.

The newly repeated random-initialization FQL baseline is not bitwise identical
to the earlier pilot: first-update logged training values match, but weights
already differ by up to 3.2e-7 after that update and diverge subsequently. Config,
update counts and final CPU/CUDA RNG states match. This is consistent with GPU
numerical nondeterminism; the historical rollout score cannot replace fresh
baseline evaluation. One-seed pilot differences require replication.

## Completed matched policy comparison

All three final checkpoints were evaluated, each on four configurations in two
complete 300-decision rounds. All 24 episodes completed without simulator errors.
No additional training or checkpoint selection followed the rollout results.

| Actor initialization | Final coverage + whole-grasp passes | Training-body passes | Withheld-body passes | Mean final coverage | Whole-grasp valid | Historical paper-filter passes | Paper filter AND whole-grasp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random / ordinary FQL | 2/8 | 1/4 | 1/4 | .29397 | 5/8 | 0/8 | 0/8 |
| Geometry pretraining | 1/8 | 0/4 | 1/4 | .40916 | 4/8 | 2/8 | 0/8 |
| Motion pretraining | 0/8 | 0/4 | 0/4 | .07112 | 3/8 | 0/8 | 0/8 |

All three coverage-plus-grasp passes also retain coverage >=.7 for the final
20 decisions. The geometry model's higher mean coverage does not imply better
valid completion: both of its paper-filter passes exceed the 2 cm grasp limit.
No motion-policy episode even reaches peak coverage .7, so final-only scoring
or the early-turn heuristic cannot by themselves explain its zero pass count.

The baseline differs from the older 3k pilot (4/8), as anticipated by the
non-bitwise training comparison. This small repeated-cell study has one training
seed and substantial variability. It supports **no improvement from this tested
actor-encoder pretraining recipe**; it does not disprove motion pretraining in
general, establish statistical superiority of random initialization, or evaluate
PointZero's original architecture/checkpoints. Only five short, mostly failing
training trajectories supplied dynamics supervision, with sparse opening queries.

| Variant | Representation pretraining | FQL training | Native evaluation |
|---|---:|---:|---:|
| Random | 0 s | 332.35 s | 211.30 s |
| Geometry | 37.32 s | 332.69 s | 174.36 s |
| Motion | 36.99 s | 332.63 s | 250.79 s |

Each learner receives exactly 3,000 FQL updates; the two pretrained variants have
additional cost. The sequential pipeline takes 1721.04 s (28.68 min) including subprocess
startup, plus 1.81 s track preparation. Smoke tests, analysis and engineering are
outside that measurement. Previously collected geometry and the RL dataset are
reused; their recorded preparation costs are disclosed above, not silently
claimed as zero-cost first-time data acquisition. Sampled learner GPU utilization
is 99–100%; peak FQL Torch allocation is about 12.03 GB.

The baseline loses three earlier high-coverage states. A post-hoc check of its
last saved opening center/normal (one decision before the final state) finds one
center 5.75 cm beyond the shoulder, consistent with the bounded reward extension
remaining a limitation. This is not a full triangle/topology reconstruction and
does not prove those episodes were physically successful. See
`baseline_peak_loss_audit.json`. No success labels or rewards were changed.

### Decision and next gate

Keep the baseline and all negative results. Do not extend this recipe to a larger
training budget on the basis of improved mean track error. The next useful test,
if continued, needs adequately represented sleeve-opening/elbow behavior and
accurate predictions of stationary contact states, followed by the same policy
comparison. Metric validation remains necessary. Continued auxiliary supervision,
critic pretraining and a larger dynamics model are untested hypotheses, not fixes
established by this pilot. No new training is running.

Artifacts: `summary.json` contains per-episode results, splits and measured costs;
`pipeline.json` and `run_pilot.py` record the sequential commands; the
`pretrain_*`, `fql_*`, and `eval_*` directories retain all checkpoints and logs.

After this run, metadata bookkeeping was clarified: future runs label
`fresh_weights=False` when importing an encoder, separately from
`fresh_rl_optimizer=True`, and derived-data manifests record the sampling seed.
The original pilot artifacts retain the older `fresh_weights` flag meaning
no full FQL resume; their explicit `encoder_initialization` records identify
transferred weights and hashes unambiguously. The pilot's sampling seed is 17.
These metadata-only clarifications do not change learning or inference. The
fourteen focused tests pass again after the clarifications.
