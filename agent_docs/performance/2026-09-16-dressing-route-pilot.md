# Dressing trajectory generation: first route pilot

Status: implemented and CPU-tested; **no native pilot run and no improvement claim**.
The GPU was occupied by the existing IAQL study (92,674 / 97,887 MiB, 99%
utilization). No study process was stopped or additional GPU job launched.

## Motivation and scope

The owner proposed using spatial-constraint demonstrations as inspiration for
making dressing policy training easier without first training 27 regional teachers.
The hypothesis is that structured strategy proposals, verified in the target
simulator, can generate useful demonstrations more cheaply. This first experiment
asks whether changing the approach route helps beyond parameter tuning. It does
not implement general strategy synthesis, recovery, imitation learning, or new RL.

References:

- [Demo methods](https://qinengwang-aiden.github.io/demos/constraint_demos/methods.html):
  recorded planned solutions with disclosed ideal grasps and reproduction limits.
- [Scaling Up and Distilling Down](https://proceedings.mlr.press/v229/ha23a.html):
  language-guided planning generates demonstrations for a shared policy.
- [ReKep](https://rekep-robot.github.io/): staged geometric constraints and feedback.
- [Interleaving Prediction, Planning, and Control](https://arxiv.org/abs/2001.09950):
  global planning, local deformable-object control, and deadlock prediction.

The [early-turn audit](2026-09-13-early-turn-filter-audit.md) locates a route defect
during `approach` / `middle`. That audit used held-out poses, so new selection uses
training poses only. It does not establish that the three training cases below
fail: the first baseline run must establish that. Existing held-out results are
diagnostic history, not fresh proof of generalization.

## Fixed comparison

Region 13, tshirt_26, training poses 0/1/2 (bodies 14000/14001/14002), 300 decisions
per episode, one environment, seed 0. Nine sequential trials:

| Arm | Three candidates | Purpose |
|---|---|---|
| baseline | z_offset=0.12 repeated three times | Fresh-world reproducibility |
| parameters | z_offset=0.12, 0.09, 0.15 m | Existing route with height search |
| route | Same heights, outward_offset=0.04 m | Matched height search with a route change |

Each arm has a cap of nine episodes / 2,700 decisions and 3 x 1,800 s elapsed
time. These are caps, not equal realized computation: report actual elapsed time,
startup and failures. Repeated baselines are not independent training seeds.
Total cap: 27 episodes / 8,100 decisions. The runner stops on the first incomplete
or invalid trial; incomplete arms cannot establish a winner.

`HeuristicDressingPolicy.outward_offset` defaults to zero. It projects the
upper-arm direction onto the plane normal to the forearm and moves away from that
inside direction. The displacement affects `approach`, `finger`, and `middle`;
later stages are unchanged. A straight arm has no selected outside and receives
zero displacement. This candidate does not guarantee clearance or success.

Existing dynamics, actions, termination and metrics remain authoritative. Report
final success / upper-arm ratio plus early-turn and paper-filter rates separately.
Do not redefine success after inspecting results. Expert manifests record selected
poses, parameter contents and environment configuration. Pilot plans record source
SHA256 hashes. Each trial starts fresh and runs continuously; no reported episode
is stitched from intermediate resets.

## Running

From the repo root:

```bash
export PYTHONPATH="$PWD/build_raw/python/src:$PWD/python"
export LD_LIBRARY_PATH="$PWD/build_raw/Release/bin:/home/ge47gax/Toolchain/uipc_cuda128/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/ge47gax/kun/genesis-world/.venv/bin/python -m uipc_manip.dressing_route_pilot \
  --out output/uipc_manip/route_pilot_plan_20260916
```

Default behavior prepares a plan without building a simulation. Use a fresh
output directory and add `--run` to execute when GPU capacity is available.
Existing directories are refused. One candidate process runs at a time; timeout
or interruption kills only that trial's process group. No job is queued
automatically. First use a fresh `--pose-ids 0 --horizon 3` run for native smoke;
that short run is not a task evaluation.

Completed trials contain `manifest.json`, `records.json`, and episode NPZs with
observations, privileged states, actions, rewards, stages and per-decision metrics.
`results.json` records exit status, elapsed time and summaries; logs and parameters
are retained. Dropped configurations, simulator errors, nonfinite final ratios,
and missing cells invalidate a trial. Failed execution exits nonzero.

NPZs remain the expert-baseline format: useful for behavior cloning and trace
inspection, but not yet the sequence-replay format for offline representation
pretraining. Terminal next-state recording / conversion and fresh action replay
remain necessary before claiming a validated pretraining corpus. Baseline repeats
are not a full action-replay validator.

## Validation and next gate

32 CPU tests passed across `test_dressing_route_pilot.py`,
`test_dressing_expert.py`, and `test_expert_baseline.py`: geometry, reset,
straight-arm behavior, split enforcement, command parsing against the existing
protocol, candidate matching, recording, partial-result rejection, and failure /
timeout handling. Native process-group cleanup and IPC behavior are not tested
by the mocked runner checks.

Next: native smoke, then the bounded comparison. If all cases are easy, expand
baseline screening on training poses before spending more search budget. If the
route helps, freeze the selected program and test unused configurations (e.g.
training poses 40-44 and a second garment), accounting for all selection attempts.
After any revision, reserve a new evaluation set. Require fresh continuous action
replay and inspect failures before adding recovery or training a student. Robustness
and reduced total cost remain open until measured across configurations and seeds.
