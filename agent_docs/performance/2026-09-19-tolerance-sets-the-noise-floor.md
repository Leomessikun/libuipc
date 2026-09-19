# The solver's convergence tolerance sets the decision noise floor — 2026-09-19

Status: first measurement complete, confirmatory run in flight. No solver code was
changed; the tolerances used here are configuration values the environment already
exposes.

The signal-to-noise record left one thing open: whether the run-to-run seed that the
dressing task amplifies is removable (`2026-09-19-decision-signal-to-noise.md`). The
two located non-deterministic reductions suggested a code change. It turns out a
configuration value does most of the work, which is worth knowing before anyone edits a
kernel.

Script: `scripts/measure_predictability_horizon.py`, which now takes
`--newton-tolerance`, `--linear-tolerance` and `--newton-max-iterations`. Artifacts:
`output/uipc_manip/tolerance_probe_20260919/{default,tight}/`.

## Protocol

One slot, `tshirt_26/14046`, scripted expert to decision 90, snapshot, then the
expert's next 20 commands replayed three times with **bitwise identical** commands from
the exactly restored state (largest restore error 0.0 m at both settings). Divergence
is the mean over run pairs of the RMS vertex distance. Only the zero-perturbation group
is used: this measures the simulator's own irreproducibility and nothing else.

## Result

| setting | newton velocity tol | CG relative tol | d1 | d5 | d10 | d20 | coverage spread | seconds |
|---|---|---|---|---|---|---|---|---|
| default | 0.1 | 1e-2 | 0.0218 | 1.0803 | 1.7971 | 2.5738 | 0.00583 | 107.7 |
| tight | 0.01 | 1e-4 | 0.0013 | 0.0031 | 0.0034 | 0.0185 | 0.00001 | 265.6 |

Separations in millimetres. Tightening both tolerances drops the separation by 139
times at decision 20 and 530 times at decision 10, and the spread of the final
upper-arm coverage by 583 times, for 2.5 times the wall time.

Read against the branch diagnostic: dressing's gap between the best macro and the next
best is 1.31 repeat standard deviations at the default tolerance. A noise floor 583
times lower puts that ratio near the control task's 193.6 — the quantity that separates
a task reinforcement learning solves from one it does not.

## What is not yet established

* **Which knob does the work.** The two tolerances moved together. Separating them is
  the next sweep.
* **That the decision margin actually improves.** A branch run at the tight tolerance
  with every other setting matched to `decision_branches_20260918/t26_14046` (seed
  3301, window 8, follow 32, three repeats, eight slots) is running; the margin will be
  read with `scripts/consequence_to_noise.py` against the default-tolerance run.
* **A clean wall-time ratio.** Both timings were taken while another job shared the
  GPU. The 2.5 times is indicative, not a benchmark.
* **That the task is unchanged.** It is not: the expert's own closed-loop coverage at
  the same decision differs (0.0368 default, 0.0483 tight), because a tighter solve is
  a more accurate one. The claim here is about the task a learner faces, and a learner
  faces whichever dynamics the tolerance produces.

## Why it matters for training

The tolerances are chosen for speed. This says what that choice costs in learnability,
and it is a trade a single workstation can make deliberately: 2.5 times the compute per
decision for a decision signal that is two to three orders of magnitude above the noise
instead of at it.

## Reproduce

```bash
python scripts/measure_predictability_horizon.py --checkpoint <ckpt> --cell tshirt_26:14046 \
  --snapshot-steps 90 --horizon 20 --repeats 3 --perturbations 0 --seed 1097 --out <dir>/default
python scripts/measure_predictability_horizon.py --checkpoint <ckpt> --cell tshirt_26:14046 \
  --snapshot-steps 90 --horizon 20 --repeats 3 --perturbations 0 --seed 1097 \
  --linear-tolerance 1e-4 --newton-tolerance 0.01 --out <dir>/tight
```
