# Contact-force audit and implications for training

2026-09-12. Base: `ab94e56b`, branch `cloth-cable-manip-rl`. The starting tree had
untracked Claude worktrees, which were preserved. This change adds Python diagnostics and
regressions; it neither changes the native solver nor establishes improved policy success.

## Decisive controlled result

One particle (radius 0.01 m, density 1000 kg/m³, mass 0.004188790204786391 kg) rests
on a half-plane under gravity (2, -9.8, 0) m/s², friction coefficient 0.5,
contact resistance 1e9. Semi-implicit early termination is disabled in both conditions.
Each run lasts two simulated seconds; measurements use the last half second. Each
configuration runs in a fresh process. This is a numerical regression, not a timing benchmark.

| dt (s) | Newton velocity tolerance | Tail Newton iterations | Mean friction x (N) | Maximum momentum-balance residual (N) |
|---|---:|---:|---:|---:|
| 1/60 | 0.05 | 1 | 0 | 0.00837758041 |
| 1/120 | 0.05 | 1 | 0 | 0.00837758041 |
| 1/60 | 1e-7 | 4 | -0.00837757784 | 2.5652e-9 |
| 1/120 | 1e-7 | 4 | -0.00837757874 | 1.6677e-9 |

Expected friction is -0.00837758041 N; expected normal support is 0.04105014401 N.
All four configurations reproduce normal support. Four additional zero-horizontal-load
controls also reproduce support, with residual below 1.4e-16 N.

The cached gradient is assembled before the last accepted Newton correction. On a
one-iteration frame, its tangential displacement can still be zero, while the accepted
position moves. Thus a zero exported friction value is not evidence that the actual
step had no frictional response. Tighter convergence fixes this particular balance test;
it does not make arbitrary contact-force exports exact final-state measurements.
The loaded particle still creeps (about 0.002 m/s) under regularized friction: this
is not a validation of history-dependent static friction or skin shear.

`F = -gradient / dt²` converts incremental-potential derivatives into **model-force
units**. Numerical consistency, contact discretization and physical accuracy remain
three different questions. The historical 321 N arm peak has not been validated by this
particle experiment. We have not rerun that dressing trajectory under tighter tolerances.

Runtime: RTX PRO 6000 Blackwell Workstation Edition, driver 595.84; Python 3.13.12,
installed pyuipc 0.0.28. No native rebuild or sanitizer run was needed for these Python
changes. Installed binary hashes, complete configs and eight summaries are archived in
[results](2026-09-12-force-calibration-results.json). Binary/source identity is not
established merely by reading this checkout's source. Raw per-frame results are under
`/tmp/codex-force-audit/regression2/` and can be regenerated:

```bash
PYTHONPATH=python /home/ge47gax/kun/genesis-world/.venv/bin/python -m pytest \
  python/uipc_manip/tests/test_contact_force_calibration.py -m cuda -q
PYTHONPATH=python /home/ge47gax/kun/genesis-world/.venv/bin/python \
  -m uipc_manip.calibrate_contact_force --gx 2 --tol 1e-7 --out /tmp/contact-new-run
```

Validation: 184 CPU tests passed (14 CUDA tests deselected); all eight new GPU matrix
cases passed. One existing CPU warning concerns an empty mean in `train_sac.py`.
The calibration regression measures default-tolerance errors rather than asserting that
an inaccurate zero-friction result must persist in future backends.

## What to borrow from IsaacIPC

[IsaacIPC](https://arxiv.org/html/2605.24339v1) separates physical and visual meshes and
maps deformation into Isaac Sim rendering. Its GMCP integrates contact samples with
geometric weights; normal-pressure patch and Hertzian tests evaluate its behavior.
Tangential traction and friction validation remain future work in its conclusion.

Our proposed use is to preserve the existing cloth simulation mesh, map its deformation
to textured visual assets, and generate randomized observations for a student policy.
A state-based teacher can collect rollouts without rendering. This is a design proposal,
not a measured training speedup. No GMCP/mortar implementation was found by a source-name
search in this checkout; it is not an existing configuration switch we can safely enable.

Before using local force maps as supervision, adopt pressure-patch and mesh-sensitivity
checks. Dividing arbitrary nodal magnitudes by a vertex area does not implement GMCP.
A rendering integration does not solve the current reward, exploration or throughput
problems, and migration to Isaac Sim has not been benchmarked here.

## Training decisions and acceptance gates

1. Establish a reproducible successful expert and unpenalized policy baseline first.
   Seed replay or behavior cloning from successful demonstrations, then evaluate on
   held-out body/garment pairs. A fresh heavy force penalty can reward avoiding contact,
   which is incompatible with dressing. This is our proposed staged curriculum.
2. Log force-channel availability and solver statistics before consuming labels.
   `contact_export_status` distinguishes missing channels from registered empty channels;
   `vertex_forces` rejects absent normal exporters and invalid data. A one-iteration
   readout deserves investigation, but iteration count alone is not an accuracy gate.
   `geometry_vertex_block` now rejects multiple instances instead of silently attributing
   only the first rigid instance's forces.
3. Collect a stricter-solver evaluation dataset and compare final motion, model forces,
   momentum balance, success and wall cost. Do not globally tighten the training solver
   based on a cheap one-particle test. Re-evaluate the historical 321 N trajectory here.
4. Once labels pass numerical and discretization checks, compare the same successful
   baseline with scalar wrench cost, distributed normal cost, then added shear cost.
   Ramp weights during fine-tuning and report success–force tradeoffs over held-out cells.
   Force can also supervise an auxiliary predictor without directly changing the reward.
5. Independent normal/tangential measurements and identified materials are required before
   claiming human safety or validated skin traction. A rigid arm surface does not model
   skin tissue deformation. Wrist-force stopping thresholds are not nodal stress limits.

No policy training was launched in this audit. The zero-success problem is not proved to
come from zero friction export, and adding IsaacIPC or force observations is not an
established fix. Existing training-throughput and watchdog evidence remains relevant;
see the September 11 infrastructure and solver-stall records.

## Literature corrections and recovered Claude tasks

The [primary-source review](2026-09-12-force-literature-review.md) supersedes claims that
all dressing work uses a wrist scalar, that an arXiv keyword search proves novelty, or
that TaCauchy's SSIM independently calibrates predicted contact force. No Free Slide
identifies discretization-induced tangential resistance even without physical friction;
Newton convergence does not remove that mechanism.

The actual Claude session JSONL and subagent records were recovered, rather than relying
on the short `/tmp/claude-4102472/response.md` clipboard export. The
[inventory](2026-09-12-claude-agent-inventory.json) retains 47 available task records from
that session, including older work. Public text outputs are preserved locally under
`/tmp/codex-force-audit/claude_agents/`; they are research notes, not verified evidence.
A final quota error does not mean that no useful intermediate work exists.

| Recovered task group | Takeover result | Work still open |
|---|---|---|
| Solver capabilities, force exports, constraints, snapshots, differentiability | Backend audit and Python guards; local force regression | Native final-state export; full differentiation implementation is not established |
| IPC force sensing, cloth contact fidelity, exact-contact tasks, contact Hessians | Primary-source IPC/No Free Slide/TaCauchy/IsaacIPC review | Frictionless mesh-slide negative control; cloth contact-patch validation; Hessian learning experiment |
| Force-aware dressing, Georgia Tech work, safety frontier | Distributed-force prior work recovered and novelty claims corrected | Controlled force-aware policy ablations |
| Skin biomechanics, injury literature, assistive-robot standards | Universal 18 N interpretation withdrawn; mechanical observables separated | Comprehensive clinical/standards review and application-specific validation |
| Beyond one sleeve, uncooperative recipients, recovery/personalization, comfort preferences | Existing completed Claude reports preserved | Source verification and task ranking remain exploratory |
| Performance, expert data, Wang audit, alternative learning pipelines | Existing repository evidence preserved | Successful held-out policy baseline; render-bridge benchmark |

Three Codex agents were used for backend/calibration, primary-source verification and
training literature. Two hit service quota limits after useful intermediate findings;
the main thread completed the calibration implementation and tests. Claude sessions
were not resumed through Claude's API, and unfinished research is not marked complete.
