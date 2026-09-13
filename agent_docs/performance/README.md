# Performance Evidence

This directory stores durable, reviewable performance conclusions. The
chronological `handoff.md` retains raw session history; accepted/rejected
experiments should gain a focused record here so future agents can distinguish
measurement from intuition.

## Evidence hierarchy

1. Correctness and safety gates first: focused tests, full aggregate simulation,
   and sanitizer coverage appropriate to the change.
2. Measure the narrow kernel/scope that the change targets.
3. Measure the enclosing subsystem and end-to-end benchmark.
4. Repeat noisy wall measurements and report variation; do not turn one run
   into a claim.

Kernel resource counts, launch counts, or attractive assembly alone do not prove
a speedup. Conversely, contact-sensitive wall time can hide a real scoped gain
when iteration counts change. Report both and state which is the primary signal.

## Records

| Record | Status | Scope |
|---|---|---|
| [Early-turn filter audit](2026-09-13-early-turn-filter-audit.md) | Measured; closes G14 | Every expert episode fails the reference filter under FMVP's own plane; the cause is the scripted `middle` stage's inner-side path, not the criterion |
| [A prior and residual RL](2026-09-13-prior-and-residual-rl.md) | Proposed; no runtime change | Why SAC from scratch is starved, what the REAL lab's DICE-RL/LPB/GMP line presupposes, the prior gate and a residual-RL mapping onto this code |
| [Recurrent pretraining proposal](2026-09-13-recurrent-pretraining-proposal.md) | Proposed; no runtime change | RLT source audit, episode replay, temporal policy/critic contracts and staged comparisons |
| [Episode-aware replay and frame history](2026-09-13-sequence-replay.md) | Implemented, opt-in; unmeasured | Episode identities, padded window sampling, rollout state and the H-frame policy behind `--sequence-replay --history-length H` |
| [RLT in the pretraining infrastructure](2026-09-13-rlt-in-pretrain.md) | Implemented, opt-in; CPU cost only | The recurrent looped transformer as the history of the actor and dense critic, per-position SAC update under its replay contract, trajectory pretraining objective, `--history-kind rlt` |
| [Force learning and elbow evidence](2026-09-12-force-training-research.md) | Historical; force proposal superseded | Primary-source survey, raw evaluation audit, and conditional experiment design |
| [2026-09-01 cross-domain main baseline](2026-09-01-cross-domain-baseline.md) | Current reference | Three-run ABD, FEM/MAS, cloth/contact envelope and synchronized stage diagnostics |
| [2026-09-08 IPC manipulation pretraining](2026-09-08-uipc-manip-pretraining.md) | Accepted | Robot cloth/cable RL environment throughput, coupler phase split, vector scaling, scripted task gate |
| [2026-09-10 dressing time step](2026-09-10-dressing-timestep.md) | Rejected as default | dt 1/30 x 3 against dt 1/60 x 6 per decision: exact Newton/PCG counts, expert ceilings on 16 live cells, friction and settle probes |
| [2026-09-10 camera rigs](2026-09-10-camera-rigs.md) | Options, measured | Stretch 3 head and gripper cameras and Wang's static arm as dressing observation modes: sources, Wang's code audit, visibility and decodability on recorded expert states |
| [2026-09-11 dressing solver stall](2026-09-11-dressing-solver-stall.md) | Bounded; the policy's slow tail is the hold, now tethered at 6 cm | Region-13 teacher stalled at 130 s per vector step: iteration anatomy of expert, random and held-push probes, contact and step-size A/B, Newton caps, the decision watchdog, a 24-environment stress gate, the checkpoint replay and the anchor tether |
| [2026-09-11 pretraining infrastructure review](2026-09-11-pretraining-infrastructure-review.md) | Analysis, levers ranked | Wall time as transitions needed times seconds each over parallelism: budget against Wang's checkpoints, time split, the Newton route, the grasp, the unused expert, contact forces |
| [2026-09-12 force training and elbow research](2026-09-12-force-training-research.md) | Research/specification; not trained | Primary-method comparison, corrected raw elbow evidence, deployment contracts and spatial-force information experiment |
| [2026-09-12 force-learning audit](2026-09-12-force-learning-audit.md) | Numerical regression passed; training proposed | Eight loaded/unloaded particle controls, cached zero-friction readout, IsaacIPC transfer ideas and recovered agent status |
| [2026-09-12 critic architecture defect](2026-09-12-critic-architecture-defect.md) | Verified defect | Our critic concatenates the action after encoding, which is the reference's rejected latent-Q baseline, worth about 0.11 in their Table I; it leaves about 0.27 of our shortfall unexplained |
| [2026-09-12 wall-clock budget](2026-09-12-wall-clock-budget.md) | Measured | Environment 63 per cent, evaluation 25.6, updates 10.3 at 35.9 ms each; the 0.31 s per transition quoted throughout is the all-in figure and the simulator costs 0.220 |
| [2026-09-12 task specification audit](2026-09-12-task-specification-audit.md) | Audit, partial | Is our task the reference's? Travel budget is three times theirs and the reward matches term for term; the grasp differs and the first tether probe found no rejections |
| [2026-09-12 research direction](2026-09-12-research-direction.md) | Direction | Force inside online simulation training, which every published dressing system names as impossible; three claims ranked by defensibility, and the fidelity ablation nobody has run |
| [2026-09-12 force training infrastructure](2026-09-12-force-training-infra.md) | Design | FCVP's force is an execution-time filter trained on 264 real trials, never in its learning loop; five components, and a three-way ablation on force fidelity that nobody has run |
| [2026-09-12 force in training](2026-09-12-force-in-training.md) | Design, gated | A force penalty is Wang's published baseline; his PCGrad ablation refutes his own diagnosis; one probe decides whether force is hidden state, and force in the critic, the world model and the curriculum survives either answer |
| [2026-09-12 contact force calibration](2026-09-12-contact-force-calibration.md) | Measured | Incremental-gradient units checked at two time steps; physical-force interpretation corrected |
| [2026-09-11 training infrastructure proposal](2026-09-11-training-infrastructure-proposal.md) | Proposal, gated | Keep IPC; privileged-state teacher seeded from the scripted expert, residual probe first; point-cloud student by Wang's teacher term on its own rollouts; Step 0 expert fix and region-13 baseline |
| [2026-08-30 case2 assembly roll-up](2026-08-30-case2-assembly-rollup.md) | Accepted with caveats | Buffer growth, collision readback, line-search aggregation, contact/FEM assembly, rejected split |
| [`handoff.md`](../handoff.md) | Historical source | Earlier MAS, CUDA graph, CUB, DyTopo, lifecycle, and detailed command history |

## Required content

Every new record should identify:

- before/after commits and whether the worktree was clean;
- GPU, driver, CUDA toolkit, build type/flags, and benchmark/submodule revision;
- scene/configuration, warmup, measured frames, repetitions, and timer semantics;
- focused, subsystem, and wall metrics with units and sample counts;
- correctness/sanitizer results and observed numerical variability;
- rejected alternatives and the decision taken;
- artifact paths or commands sufficient to reproduce the run.

Use [0000-evidence-template.md](0000-evidence-template.md). Canonical large
scenes are launched through `benchmarks/manifest.json` and
`scripts/run_benchmark.py`; its archived metadata supplies the minimum runtime
and revision facts. Approved regression thresholds belong in versioned
`uipc.profile` baseline artifacts with environment compatibility checks.
The 2026-09-01 shared-WDDM baseline is deliberately a measured range rather
than an enforced threshold; promote it only on a stable dedicated runner.

## Interpretation rules

- Do not add percentages from different stages as if they were one controlled
  experiment.
- Prefer median plus distribution/repeats for wall time; keep raw means when
  comparing to historical reports that used means.
- Parent timers include children unless explicitly documented otherwise.
- A different Newton/PCG/contact count is an algorithmic-path change, not pure
  throughput evidence.
- Atomic-order trajectory drift must be compared with unchanged-run variability,
  not assumed to be either a bug or harmless.
- Record regressions and rejected designs; they prevent expensive repetition.
