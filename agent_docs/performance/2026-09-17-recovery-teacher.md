# IPC recovery teaching: novelty audit and bounded experiment

## Research claim and prior work

The owner asked whether the proposed recovery-teaching idea is novel and, if
promising, to try it. The broad recipe is established, not a new RL contribution:

| Primary source | Overlap with our proposal |
|---|---|
| [DAgger, 2011](https://proceedings.mlr.press/v15/ross11a.html) | Aggregate expert supervision at states induced by the learner to address distribution shift. |
| [Guided Policy Search, 2013](https://proceedings.mlr.press/v28/levine13.html) | Use trajectory optimization to guide policy learning. |
| [ThriftyDAgger, 2021](https://arxiv.org/abs/2109.08273) | Query intervention selectively under a budget; includes physical cable routing. |
| [Safe Learning of Locomotion Skills from MPC, 2024](https://arxiv.org/abs/2407.11673) | Iteratively learn from an MPC expert with SafeDAgger-inspired intervention. |
| [MPC Scaffolding for Dexterous Manipulation, v2 16 September 2026](https://arxiv.org/html/2609.14878v2) | Pretrain an actor/critic from MPC trajectories, retain experience, and guide online SAC intermittently. This is a particularly close framework precedent. |

Potential contribution: demonstrate that expensive, accurate IPC simulation can
provide useful, affordable recovery supervision for cloth dressing, with explicit
checks that a selected correction survives the full continuation and actually
improves the learned policy. This is a hypothesis about a mechanism and its
empirical value, not a verified novelty claim. A reliable robotics system or
careful negative result can be a contribution without a new SAC update. Neither
the use of IPC nor a renamed auxiliary imitation loss establishes novelty.

This pilot reuses existing geometric experts; it does not invent a new trajectory
optimizer. If a fixed expert works equally well, that weakens any claim that
search/verification was necessary. A later positive claim needs comparison with
ordinary corrective imitation, unverified expert labels, equal-budget search,
and a lower-cost simulator, plus multiple training seeds, bodies and garments.
Real-world performance remains untested.

## Why this experiment differs from the earlier failed short search

The earlier CEM search optimizes 12 commands and uses a 60-decision continuation;
its small search-score gain did not survive validation. The new test starts at
a state visited by the actual expert-pretrained actor and evaluates closed-loop
route controllers through decision 300. It cannot accept a transient peak such
as the previously observed .99-to-zero coverage collapse.

The source actor is
`output/uipc_manip/expert_pretrain_20260917/bc/checkpoints/actor_final.pt`.
Only the BC-training-body configuration tshirt_68/14046 is used to create
candidate labels. Bodies 14047 and 14048 remain withheld from actor fitting.
Approach: 120 closed-loop BC decisions, selected from the previously observed
failed configuration. This is a fixed development-state pilot, not an implemented
learned failure detector or a general policy for when to request IPC computation.

## Fixed protocol

Two physical cloth copies share a batched IPC world. The six controllers are:

1. Current actor under the common command caps.
2. Actor with translation scaled to the common cap (speed-only control).
3. Existing expert resumed at the middle/forearm stage.
4. Existing expert resumed at alignment.
5. Existing expert resumed at elbow-hook.
6. Middle-stage expert with the existing .04 m outward-route option.

Every controller visits every saved slot using the existing cyclic scheduler.
Each candidate is closed-loop, with its own fresh expert stage/counters. Action
translation is capped at 8 mm/decision and effective rotation at .05 rad/decision;
x rotation is inactive. These are equal upper budgets, not identical executed
travel. The scaled-policy control and actual commanded/accepted distances expose
speed confounding. The learner's approach retains its normal commands.

Each branch runs all remaining 180 decisions. Score is the minimum upper-arm
coverage in the last 12 decisions, then the minimum across the two slots. Any
whole-episode grasp violation or simulator error invalidates a route. The
highest-scoring expert route is frozen after search. If no expert reaches .7
sustained coverage on every slot, the experiment stops without training.

A new world with reset seeds advanced by 1,000 verifies that frozen route versus
both controls, with reverse control order and balanced slot assignment. Admission
requires all verification copies to have at least .7 sustained coverage, maximum
tracking at most .02 m including the learner prefix, and at least .05 sustained
coverage improvement over **both** controls at the corresponding slot. This is a
conservative empirical gate, not a statistical guarantee from two samples.

Only admitted verification trajectories can have `kept=True` in the distillation
dataset. Search trajectories and control actions remain unkept. Observation is
recorded before its corresponding action; privileged geometry is logged but not
used as a student input. Historical early-turn/paper-filter results remain
separate: this gate does not certify a successful episode under that definition.

If admitted, the next fixed comparison is two copies of the same BC actor,
1,000 additional updates each: old demonstrations alone versus old demonstrations
plus admitted recovery sequences. Evaluate the fixed final actors against the
unchanged BC actor on tshirt_26/14046, tshirt_68/14046, tshirt_26/14047 and
tshirt_68/14048, two rounds each (24 full episodes total). Use seed 1, batch 128,
MSE, learning rate 1e-4 and complete validation every 500 updates. Uniform sampling
of the combined training rows gives the recovery examples their natural data
fraction; no adaptive weights or checkpoint selection are added. No training extends the
budget or selects a checkpoint from physical evaluation. If admission fails,
there is no justified correction-labelled policy training in this pilot.

## Implementation and verification

Base commit `9ac7b0fa`, current branch `research/ipc-adjoint-q-learning`; preexisting
untracked build/worktree files preserved. `uipc_manip.recovery_teacher` reuses the
existing heuristic, `EpisodeTape`, branch snapshot/restore and cyclic assignment.
The environment gains optional `step(..., reset_on_done=False)` to preserve the
true time-limit state after recording terminal observations. Default automatic
reset is unchanged; simulator errors still reset. This avoids extending the
episode horizon or accidentally collecting observations from the next episode.

Seventeen focused tests pass for admission, command caps, existing parallel search
and distillation. `distill --init-actor` initializes only actor weights, checks the
saved protocol and leaves critic/optimizer state fresh; its behavior is tested.
Two native CUDA tests pass for both automatic reset and retained terminal state.
Implementation commits: `5272f35e` (teacher) and `bcc44472` (actor-only continuation).
Native search, verification, both fixed actor continuations, the 24-episode
student comparison, and the separate 12-episode cap diagnostic have completed.
The later owner-requested [SAC integration](2026-09-17-recovery-sac-pretraining.md)
is a separate experiment; none of the results below includes a SAC learning update.

## Completed teacher and training results

| Search controller | Sustained valid successes | Minimum late coverage over both slots |
|---|---:|---:|
| Policy | 0/2 | .00000 |
| Faster policy | 0/2 | Rejected: grasp violations |
| Middle-stage expert | 1/2 | .00000 |
| Alignment-stage expert | 1/2 | Rejected: one grasp violation |
| Elbow-hook expert | 0/2 | .37292 |
| Outward-route expert | 2/2 | .99037 |

The middle-stage expert reaches .99402 maximum coverage on its failed slot but
finishes at zero. The elbow-hook expert finishes one slot at .99371, yet its
last-12-decision minimum is .37292. These are distinct reasons for rejecting
peak-only and last-frame-only teaching criteria in this pilot.

In fresh-world verification, the frozen outward route achieves sustained
coverage .98444/.97604 and final coverage .98481/.97880 with valid grasps.
Policy and scaled-policy controls both have zero sustained coverage on both
slots. The policy retains valid grasp; the scaled policy violates it. The
admission gate passes, producing two 180-decision recovery sequences (360 rows).
Both admitted sequences still fail the historical early-turn/paper filter.

Search costs 2,400 native decisions / 351.24 s; verification costs 1,320 / 221.97 s.
Totals: 3,720 decisions / 573.21 s, including both world constructions and policy
approaches. All reported position restore errors are zero; this does not imply
complete numerical trajectory repeatability or independent convergence between
batched copies. No simulation errors occurred.

The control continuation trains on 1,800 original rows; the recovery continuation
trains on 2,160 rows, of which 1/6 are new recoveries. Both use the same 1,800
withheld-body validation rows. Each completes 1,000 updates in 29.1 s through
scheduled final validation, with no simulation during updates. This excludes
process startup, loading and final extra validation/checkpoint writes. Final
validation MSE is .030163 for the control and .035575 for recovery training;
no policy benefit is inferred from these prediction errors.

## Completed closed-loop student comparison

| Actor | Final coverage/grasp successes | Mean final coverage | Withheld-body successes | Invalid-grasp decisions |
|---|---:|---:|---:|---:|
| Original BC | 2/8 | .26551 | 0/4 | 0/2,400 |
| Another 1,000 original-data BC updates | 2/8 | .27350 | 1/4 | 263/2,400 |
| Another 1,000 BC updates including recovery data | 3/8 | .38375 | 1/4 | 130/2,400 |

Every geometric success also has valid grasp throughout its episode; sustained
last-12-decision successes have the same counts. All three actors fail the
target tshirt_68/14046 in both rounds. Recovery BC retains tshirt_26/14046 in both
rounds and succeeds on tshirt_26/14047 once. Continued BC loses a previously
successful training case in one round. Controller collision rejections total
0 / 1,719 / 0, respectively; no tether rejection or simulation error occurs.
Historical paper-filter passes are 0/8, 1/8 and 0/8, respectively.

The 24 complete episodes cost 7,200 native decisions / 648.66 s. These are two
repeats of four development configurations, not independent training seeds or an
untouched test set. One extra success with substantial rollout variation does
not establish a robust policy gain or successful transfer of the target recovery.

Read-only evaluation on the same 360 teacher observations gives active-action
MSE .12874 / .12387 / .01281 for original/continued/recovery BC. Recovery BC fits
the recorded teacher actions about ten times more closely, yet still fails the
target rollout. Fitting error on recorded observations is therefore insufficient
as a policy-quality measure. Distribution shift in the learner's approach and
subsequent feedback remains a plausible cause, not an isolated diagnosis.

### Additional command-limit diagnostic

After the first full comparison round, the recovery actor still fails the target
sleeve and exceeds the teacher's 8 mm/decision cap on 184/300 decisions, reaching
11.57 mm. Its coverage at decision 120 is .13145 versus the unchanged actor's
.19918: actor fitting has already changed its approach to the correction state.
The first grasp violation is at decision 238. These observations motivate a
separate inference-only cap ablation; they do not establish the cause by themselves.

After the fixed two-round comparison, evaluate all three unchanged checkpoints
for **one additional round** on the same four configurations, now applying the
teacher's 8 mm translation/.05 rad rotation norm caps to all actors. This is an
explicit follow-up diagnostic chosen after observing first-round failures,
not a replacement for the primary results or another training/checkpoint search.
`evaluate_dressing_policies.py` records the optional caps and executed commands.
The analysis verifies every measured command stays within the specified limits.
Original/continued/recovery BC achieve 1/4, 0/4 and 1/4 valid-grasp successes;
mean final coverage is .39602, .13465 and .24744. Invalid-grasp decisions are
0, 274 and 58. All actors still fail the target sleeve. The 12 episodes cost
3,600 decisions / 305.47 s including setup. This exploratory one-round check
does not rescue the recovery policy or prove the command limit is the sole cause.

## Conclusion and accounting

IPC found and independently verified a useful recovery controller on one failed
configuration. This particular actor-only imitation recipe has not transferred
that recovery into a robust autonomous policy. Do not merely extend its BC
budget or present it as an improved SAC algorithm. The owner subsequently asked
to integrate the guidance into actual RL pretraining; that work is recorded
separately and must be assessed on its own results.

All jobs in this experiment finished. Teacher search/verification plus primary
and cap evaluation total 14,520 native decisions / 1,527.35 s (25.46 minutes).
The two actor fitting loops add about 58.2 s through scheduled validation;
startup, final extra validation and checkpoint I/O are outside that training
scope. Existing demonstrations and the source actor's training cost are shared
inputs, not free data. Artifacts include `analysis_summary.json`,
`prediction_audit.json`, `teacher_comparison.png`, and `student_comparison.png`.

Hardware/runtime: RTX PRO 6000 Blackwell Workstation Edition, driver 595.84,
native Release build with CUDA 12.8, PyTorch 2.12.0+cu130. Native physics is batched
on CUDA; route logic, snapshots and observation/metric handling still involve the
CPU. No solver or SAC update changes, no additional C++ build/sanitizer run.

```bash
env PYTHONPATH=build_raw/python/src:python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LD_LIBRARY_PATH=build_raw/Release/bin:/home/ge47gax/Toolchain/uipc_cuda128/lib \
  /home/ge47gax/kun/genesis-world/.venv/bin/python -m uipc_manip.recovery_teacher \
  --checkpoint output/uipc_manip/expert_pretrain_20260917/bc/checkpoints/actor_final.pt \
  --out output/uipc_manip/recovery_teacher_20260917 \
  --cell tshirt_68:14046 --approach 120 --slots 2 --seed 2197
```

Output paths must be fresh. `search/result.json` contains all route/slot outcomes,
work counts, timing, restore checks and full episode tapes. If selection passes,
`verification/` holds the independent execution and distillation manifest/records;
root `result.json` records the admission decision. No native test is left running
as a prerequisite after a failed gate.

Actor reproduction, after a successful teacher admission (fresh output run names
are required):

```bash
export PYTHONPATH=build_raw/python/src:python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=build_raw/Release/bin:/home/ge47gax/Toolchain/uipc_cuda128/lib
PY=/home/ge47gax/kun/genesis-world/.venv/bin/python
ROOT=output/uipc_manip/recovery_teacher_20260917
BASE=output/uipc_manip/expert_pretrain_20260917/bc/checkpoints/actor_final.pt
DATA=output/uipc_manip/expert_pretrain_20260917/dataset
$PY -m uipc_manip.distill --source-dirs "$DATA" \
  --teacher-checkpoint "$BASE" --init-actor "$BASE" \
  --validation-bodies 14047 14048 --preload-to-device \
  --work-dir "$ROOT/learning" --run-name continued_bc \
  --steps 1000 --batch-size 128 --loss mse --eval-every 500 --save-every 0 --seed 1
$PY -m uipc_manip.distill --source-dirs "$DATA" "$ROOT/verification" \
  --teacher-checkpoint "$BASE" --init-actor "$BASE" \
  --validation-bodies 14047 14048 --preload-to-device \
  --work-dir "$ROOT/learning" --run-name recovery_bc \
  --steps 1000 --batch-size 128 --loss mse --eval-every 500 --save-every 0 --seed 1
$PY scripts/evaluate_dressing_policies.py --reference "$BASE" \
  --policy "original_bc=$BASE" \
    "continued_bc=$ROOT/learning/continued_bc/checkpoints/actor_final.pt" \
    "recovery_bc=$ROOT/learning/recovery_bc/checkpoints/actor_final.pt" \
  --cells tshirt_26:14046 tshirt_68:14046 tshirt_26:14047 tshirt_68:14048 \
  --rounds 2 --out "$ROOT/evaluation.json"
```

The separate cap diagnostic uses that evaluation command with `--rounds 1`,
`--translation-cap-m .008 --rotation-cap-rad .05`, and
`--out "$ROOT/command_cap_evaluation.json"`. The actor weight files are identical
in the primary and capped comparisons. Artifact `analyze_results.py` generates
the teacher/student plots and aggregates; `audit_student_predictions.py` measures
all three fixed actors on the same demonstration/recovery observations without
optimization. Neither script selects a new checkpoint.
