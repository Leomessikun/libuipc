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
| [IAQL direct-picker benchmark](2026-09-14-iaql-benchmark.md) | Mechanics validated (raw Hessian on 100 states, friction 0.6 with the lagged coupling); the 5k-step SAC/IAQL pair is inconclusive because neither arm learns the task at that budget | Paired SAC value/action-gradient targets, complete-decision x/v tangent, matched SiLU networks: held-out slope cosine 0.38 → 0.84 with paired labels and none with shuffled; on 100 frictionless states the solver's SPD-projected matrix gives a 4.3 % median tangent error growing with drag (1.6 → 8.1 %) while the raw Hessian re-assembled at the accepted state gives 0.20 % flat in drag (re-assembly alone changes nothing), so the projection was the whole error; friction 0.6 failed the gate 1/4 for want of the lagged friction terms; with the raw Hessian and the exported lagged coupling it passes 3/4 at 0.14 % median tangent error (the rejected state is the critic's) |
| [RLT infrastructure review](2026-09-13-pretraining-infrastructure-review.md) | Audited and repaired; no policy gain measured | Context-consistent learning, causal pretraining, corrected cost evidence and research priorities |
| [Offline representation pretraining](2026-09-13-offline-pretraining.md) | Implemented; toy corpus only | `pretrain_offline` trains the actor's encoder (and RLT history) on recorded episodes with an episode split and a constant-predictor baseline; `--init-representation` adopts it in a fresh run |
| [Physics gradients as control signals](2026-09-13-physics-gradients.md) | Level 1 probe: 3 cells, 7 states, rotation at 2 elbows, ~75 GPU min; Level 2 adjoint: 4 states, ~50 GPU min | Finite-difference sensitivity of coverage, contact energy and arm force to the gripper command at the expert's elbow, passed and stall states: after the elbow the coverage gradient is a repeatable local derivative and walking it beats the expert; at the elbow the coverage reward is flat in all five dimensions, but the 5-D axis-proxy gradient (translation + rotation) walks one of two quiet elbows over (+0.077 coverage, 3× the expert) and not the other; one stall is an environment lock (no-move rule, 12.6 vs 12 mm) that release-then-advance escapes. Level 2a: the adjoint through the solver's own Hessians (BDF1 chain over a decision's six frames, soft position constraint as the command's entry) reproduces the smooth objective's differences to cosine 0.989–0.995 at four states and the whole free-cloth response field within 25 % where contact is light, not at a jam; `thickness` is a half-thickness in libuipc. Level 2b: `LinearSystemAdjointFeature` exports the frame's assembled system and solves against it with refined PCG (2e-7 in 0.15 s), bit-identical to the debug dump; `uipc_test_diff_sim`. Level 3 first experiment: open-loop 12-decision trajectory optimisation with the 72-frame chain walks both quiet elbows over (coverage 0.167 and 0.106 vs the expert's 0.028), but the chain reduces to a static sensitivity beyond one decision and most of the gain is the proxy direction at the action box (0.162 / 0.094 without any gradient); the actor that fits is SVG(1)-like (one-decision Jacobian × TD critic). Its pieces checked: a trained dense critic's ∂V/∂x' pushed through the one-decision adjoint, followed greedily for 12 decisions, walks both elbows over (coverage 0.168 / 0.135) where the same critic's ∂Q/∂a at the same hold action retreats (0.000 / 0.000; it wins only after the elbow, 0.184 vs 0.151), its policy and the expert do not, and beats the hand proxy at the elbow it could not solve; a 14k-update latent critic gives the adjoint nothing to carry; V is not smooth at 2 mm (differences 20–30× the derivative, a factor 2–7 of it the observation's voxel/visibility switches). Fine-tuning the actor on 288 elbow transitions with the critic frozen (the tuned actors sit on the action box: the loss is linear in μ): the sign pattern the physics direction teaches passes both elbows (0.158 / 0.128) and is the gentler and better one at both states that were not tuning start states (0.222 at 18 N vs SAC's 0.168 at 318 N; 0.189 vs 0.116); SAC's own loss is stronger only at cell 3's elbow and turns cell 1's into a retreat. Not a training run |
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
