# Existing-expert dressing policy pretraining — 2026-09-17

## Purpose and current result

Train the visual actor from existing successful expert actions, avoiding an IPC
simulation step for every gradient update. This implements the owner's approved
pretraining step. It is ordinary behavior cloning (BC), not a new RL algorithm
or an IPC-gradient modification to SAC. Completed evaluation gives BC 2/8
coverage-and-grasp successes versus existing SAC 0/8. Both BC successes are the
same training configuration, repeated; withheld bodies score 0/4. Mean coverage
is lower for BC (.25968 versus .29953). Both policies pass the historical
early-turn filter 0/8. Fast pretraining works; robust dressing remains unsolved.

Base commit: `387e1132`, implementation `426b43a2`, branch
`research/ipc-adjoint-q-learning`. Existing
untracked build/worktree/output files were present and preserved. Changes reuse
`uipc_manip.distill`, `EpisodeTape`, the saved actor architecture and the existing
matched dressing evaluator. No solver or SAC update equation changed.

## Corpus audit and admission

The corpus is larger than the earlier 25-episode IPC diagnostic subset. A metadata
scan of `/home/ge47gax/kun` found substantial existing Newton demonstration and
replay collections. File counts include individual diagnostic frames and must
not be described as counts of independent demonstrations.

One inspected Newton buffer,
`newton/artifacts/dressing_physics_v5_demos_tshirt26_h7_n36/replay_buffer.pt`,
declares 36 successful episodes and contains 12,857 transitions. It uses a 2,005D
x-ray observation, a different cloth solver and control/grasp contract. This
is not directly interchangeable with the current 5,383D IPC observation.
Reusing it requires explicit observation/action conversion and a transfer study.
It was not mixed into this baseline. Two native observation-equipped smoke
episodes also use old physics and have poor grasp/coverage; they were excluded.

The current native source `output/uipc_manip/expert_r13_heldout_s0` stores saved
actions and 35D geometry summaries, but no policy observation sequence. Six of
its 25 episodes have final upper-arm coverage at least .7, maximum tracking error
at most .02 m and no simulator error. The reconstruction script replays those
six saved action sequences twice, records observation **before** the associated
action, and grades each new execution afresh. It does not generate new expert
commands or inherit the historical success label.

All 12 reconstructed episodes pass this explicitly named
`geometric_and_grasp_v1` rule: mean final coverage .96398, maximum tracking across
the set .01535 m, 300 decisions each, zero simulator errors. All 12 still trigger
the historical early-turn flag and fail `paper_filter`; the new admission rule
is a deliberate experimental distinction, not a claim of paper-filtered data
or a comprehensive physical safety certification. Both labels are preserved.
Original paths, source records and file SHA-256 hashes accompany each replay.

Reconstruction used two three-cell native worlds, two repeats per world, and
3,600 decisions in 318.63 s including construction and output writes. The result
is cached; subsequent BC runs need no simulation to consume it. Replaying twice
does not create twelve independent expert strategies.

## Training and split

| Partition | Garment/body configurations | Episodes | Transitions |
|---|---|---:|---:|
| Train | tshirt_26/14046, tshirt_68/14046, tshirt_4/14046 | 6 | 1,800 |
| Validate | tshirt_26/14047, tshirt_68/14047, tshirt_68/14048 | 6 | 1,800 |

All repeats of a body stay in one partition. Body 14046 was historically a
development/held-out configuration; using it for this BC changes that status.
The validation bodies are withheld from **this BC run**, not pristine unseen
research test configurations. The current training set covers only one body.

The actor starts randomly, using the exact architecture saved in the current
SAC checkpoint: Wang-flow, PointNet2, hidden size 1024, plain trunk. The CLI
`--teacher-checkpoint` supplies architecture only; it does not copy actor weights.
The critic is untrained and has no role in BC. Do not resume this checkpoint as
a fully trained SAC agent without explicitly initializing the critic/training
state for the next stage.

Fixed budget: seed 1, 3,000 updates, batch 128, Adam at 1e-4, MSE on bounded
expert actions, gradient clipping at 10. Dataset tensors and random batch indices
reside on CUDA. The existing trainer now avoids per-update scalar readback and
uses fused CUDA Adam. It checks all validation transitions every 500 updates.
The final 3,000-step actor was selected for physical evaluation before results;
`actor_best.pt` is retained but not selected using rollout performance.

The loop through the last scheduled validation took 86.8 s, about 34.6 updates/s
or 4,424 sampled training rows/s, including periodic validation/checkpoint work.
This excludes process startup, dataset loading and the final extra evaluation
and checkpoint save. It is one measurement, not a controlled trainer speedup
comparison. There are zero environment steps during actor training.

Training minibatch MSE changes from .097205 to .003660; the complete withheld-body
validation MSE changes from .099883 after update 1 to .044945 at update 3,000.
Validation remains appreciably worse than training; extending optimization alone
is not justified by this curve. Loss values do not measure dressing success.
A read-only final checkpoint audit gives full-training MSE .003948 and validation
MSE .044945. A constant mean action fitted on the training set gives .081636 and
.089284 respectively. The network learned predictive structure beyond a constant
action, but this has not generalized to closed-loop success on withheld bodies.

## Full dressing evaluation protocol

Evaluate the existing 127,416-transition SAC checkpoint and the fixed final BC
actor on tshirt_26/14046, tshirt_68/14046, tshirt_26/14047 and tshirt_68/14048.
Each policy gets two full 300-decision rounds, eight episodes per policy, using
the same environment contract and reset seeds. Reverse policy order in round 2.
Report BC-training-body and BC-withheld-body groups separately, plus coverage,
final and whole-episode grasp checks, controller rejection and simulator errors.
This costs 4,800 native decisions if all episodes reach their horizons.

The pre-existing SAC and freshly trained BC have different training histories;
this answers whether this pretraining recipe produces a useful policy at its
measured cost, not which algorithm wins with matched total data/compute. Neither
result establishes an IPC-vs-other-solver advantage.

## Completed full-episode results

| Policy / BC split | Episodes | Coverage-and-grasp successes | Mean final coverage | Whole-episode valid grasp |
|---|---:|---:|---:|---:|
| SAC / all | 8 | 0 | .29953 | 2/8 |
| Expert BC / all | 8 | 2 | .25968 | 8/8 |
| SAC / training body | 4 | 0 | .26818 | 0/4 |
| Expert BC / training body | 4 | 2 | .51935 | 4/4 |
| SAC / withheld bodies | 4 | 0 | .33087 | 2/4 |
| Expert BC / withheld bodies | 4 | 0 | .00000 | 4/4 |

The two successful BC episodes are tshirt_26/14046, final coverage
.96253/.98436. The training configuration tshirt_68/14046 finishes at
.13052/.00000. In its second round it temporarily reaches approximately .99
before coverage abruptly drops to zero, while grasp remains valid. The trace
alone does not establish whether cloth geometry or the coverage metric explains
this discontinuity; maximum coverage must not be substituted for final success.
Both withheld configurations have zero upper-arm coverage throughout their BC
episodes. Thus the shortfall includes generalization before upper-arm progress
and a separate loss of progress; it is not established as one elbow-only cause.

SAC has 109 invalid-grasp decisions out of 2,400; BC has zero. Both policies have
zero recorded collision/tether command rejections and zero simulator errors.
BC makes accepted motions with valid grasps even on failed episodes: rejected
commands and lost grasp do not explain those failures. All 16 evaluation
episodes fail the historical early-turn paper filter. The BC successes therefore
apply only to the explicitly reported coverage/grasp definition.

All 4,800 evaluation decisions completed in 454.46 s including 6.65 s construction.
SAC rounds take 146.76/142.35 s and BC rounds 78.02/79.73 s. Their physical paths
differ, so the latter is not an actor inference or controlled solver speedup.
Reconstruction, training-loop and evaluation scopes sum to 859.90 s, excluding
training startup/final writes and the small prediction audit. The 86.8 s BC
figure must not be described as end-to-end dataset preparation and evaluation.

The next learning step should increase compatible body/state coverage and teach
recovery at the learner's actual failed states. Repeated optimization on the same
one-body successes has no demonstrated route to robustness. Audit the coverage
collapse and admission disagreement before using them as new improvement labels;
require any IPC recovery teacher to improve full continuations before fitting
those targets. This is a proposed next step, not an implemented recovery method
or a reason to launch another long SAC run. No additional training was started.

## Validation, environment and reproduction

Seventeen focused CPU tests pass, covering body/repeat separation, independent
validation size, observation/action alignment, excluded failed reconstructions,
incompatible physics with equal tensor sizes, and existing collection/expert
protocol behavior. A final loader fix ensures an optional coverage threshold can
only tighten the explicit admission rule, never re-admit a rejected episode or
silently substitute the paper filter. The measured run did not use that optional
threshold, so its dataset and results are unchanged. Native reconstruction,
all 3,000 CUDA updates, full evaluation and checkpoint prediction audit completed.

Machine: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB reported
memory, driver 595.84. Native `build_raw` is Release with CUDA 12.8; PyTorch is
2.12.0+cu130. No C++ changes or sanitizer run in this stage. Native runtime uses:

```bash
export PYTHONPATH=build_raw/python/src:python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=build_raw/Release/bin:/home/ge47gax/Toolchain/uipc_cuda128/lib
PY=/home/ge47gax/kun/genesis-world/.venv/bin/python
REF=output/uipc_manip/dressing_redesign_20260916/warm_sac/checkpoints/checkpoint_00127416.pt
RUN=output/uipc_manip/expert_pretrain_20260917

# Output directories/files must not already exist; choose a fresh RUN to repeat.
$PY scripts/reconstruct_dressing_demonstrations.py \
  --source output/uipc_manip/expert_r13_heldout_s0 --reference "$REF" \
  --out "$RUN/dataset" --batch-size 3 --repeats 2
$PY -m uipc_manip.distill --source-dirs "$RUN/dataset" \
  --teacher-checkpoint "$REF" --validation-bodies 14047 14048 \
  --preload-to-device --work-dir "$RUN" --run-name bc \
  --steps 3000 --batch-size 128 --loss mse --eval-every 500 --save-every 0 --seed 1
$PY scripts/evaluate_dressing_policies.py --reference "$REF" \
  --policy "sac=$REF" "expert_bc=$RUN/bc/checkpoints/actor_final.pt" \
  --cells tshirt_26:14046 tshirt_68:14046 tshirt_26:14047 tshirt_68:14048 \
  --rounds 2 --out "$RUN/evaluation.json"
```

Artifacts under `output/uipc_manip/expert_pretrain_20260917/`: `corpus_audit.json`,
`dataset/{manifest,episode_metrics}.json`, `dataset/episodes/*.npz`,
`bc/config.json`, `bc/distill_log.csv`, `bc/checkpoints/actor_final.pt`, and
`evaluation.json` with per-decision actions and controller/grasp traces.
`results_summary.json` holds all episode/group results; `comparison.png` plots
the full trajectories; `prediction_audit.json` verifies final prediction error
against constant baselines. Re-run `analyze_results.py` and `audit_prediction.py`
in that artifact directory for their read-only analyses. All jobs finished.
