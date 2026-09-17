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
unchanged BC actor, on familiar and BC-withheld bodies. No training extends the
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

Sixteen focused tests pass for admission, command caps, existing parallel search
and distillation. Two native CUDA tests pass for both automatic reset and retained
terminal state. The native teacher run is in progress; no correction-guided actor
has been trained and no useful recovery has yet been established.

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
