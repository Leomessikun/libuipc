# How predictable is garment-arm contact? — 2026-09-18

Status: completed measurement on the native IPC dressing environment. No policy was
trained and no solver, reward or controller setting was changed. It answers a
question this project kept assuming an answer to: how long a continuation from a
known state is worth predicting, and how large a command difference has to be before
its consequence can be told apart from the simulator's own irreproducibility.

Scripts: `scripts/measure_predictability_horizon.py` (collection) and
`scripts/analyze_predictability_horizon.py` (growth rates). Artifacts:
`output/uipc_manip/predictability_20260918/{t26_14046,t392_14046}/`
(`result.json`, `traces.npz`, `horizon.json`).

## Protocol

One slot, driven by the scripted expert to a snapshot decision, then:

1. the expert's next 40 commands are recorded in closed loop, giving a fixed command
   sequence;
2. the world is restored and that **identical** sequence is replayed five times;
3. the same sequence is replayed five times each with a perturbation of
   1e-4, 1e-3 and 1e-2 normalized units added to the **first** command only, that is
   0.87, 8.7 and 87 micrometres of commanded tool translation against a per-decision
   limit of 8.66 mm.

Every decision records all cloth vertex positions, the opening centroid, the
upper-arm coverage ratio and the grasp tracking error. Divergence is the mean over
run pairs of the RMS vertex distance. Cells: `tshirt_26/14046` (snapshots at
decisions 40, 90, 140) and `tshirt_392/14046` (40, 90, 140, 200), the garment whose
sleeve no controller here has ever threaded. Cost: 2,660 decisions / 18.0 min and
3,560 decisions / 20.7 min, both complete with no simulator error.

**Restore is exact.** The largest vertex position error over 63 and 84 restores is
0.00e+00 m, so nothing below depends on a lossy snapshot.

## Result 1: identical commands from an identical state do not reproduce

| Cell, snapshot | Stage | RMS separation at decision 1 / 5 / 10 / 20 / 40 (mm) | Growth per decision | Doubling |
|---|---|---|---|---|
| t26, 40 | middle | 0.16 / 0.70 / 1.64 / 3.06 / 18.47 | 0.076 | 9.1 decisions |
| t26, 90 | middle | 0.18 / 7.65 / 6.01 / 6.19 / 21.26 | 0.074 | 9.4 |
| t26, 140 | elbow_hook | 0.75 / 2.12 / 2.25 / 2.14 / 2.08 | 0.004 | 184 |
| t392, 40 | middle | 0.05 / 6.99 / 105.73 / 93.57 / 98.89 | 1.258 | 0.6 |
| t392, 90 | middle | 0.03 / 4.83 / 10.35 / 8.38 / 6.73 | 0.047 | 14.9 |
| t392, 140 | align_pitch | 0.14 / 0.92 / 2.05 / 21.33 / 18.43 | 0.099 | 7.0 |
| t392, 200 | elbow_hook | 0.03 / 13.38 / 19.99 / 12.21 / 8.99 | 0.023 | 30.3 |

The source is the simulator's own run-to-run non-determinism, since the commands are
bitwise equal and the restored state is exact. Its amplification is what the table
measures: a separation that starts below a tenth of a millimetre reaches millimetres
within 10 decisions (1 s) and centimetres within 40 at most states.

## Result 2: the perturbation's size does not matter

At every snapshot, the four groups (0, 1e-4, 1e-3, 1e-2) give the same separation
curve within their own scatter; the table in `horizon.json` shows no ordering by
perturbation size at decisions 10, 20 or 40. A deliberate 87-micrometre command
change is therefore indistinguishable, after ten decisions, from changing nothing at
all.

**Consequence.** There is a resolution limit on control: two commands that differ by
less than about one percent of a decision's translation have consequences this
simulator cannot separate, no matter how exact its derivatives are. A gradient step
smaller than that limit is not a physically meaningful action change, and a finite
difference taken at that scale measures the noise floor. This is the mechanism
behind the earlier finding that the physics gradient's useful horizon is about one
decision (`2026-09-13-physics-gradients.md`) and behind the failure of the
IPC-labelled actor updates (`2026-09-14-iaql-benchmark.md`).

## Result 3: the microstate is chaotic, the task variable is not

At decision 40 of `t26`'s snapshot 40, 98.0 % of garment vertices sit more than a
millimetre apart between two runs of the identical commands, median 17.8 mm,
maximum 31.3 mm; for `t392` snapshot 40 the median is 60.9 mm and the maximum
252.7 mm. The whole garment is in a different configuration.

Over the same runs:

| Cell, snapshot | Coverage-ratio spread at decision 5 / 10 / 20 / 40 | Opening-centroid separation at 40 (mm) | Tracking-error spread at 40 (mm) |
|---|---|---|---|
| t26, 40 | 0.000 / 0.000 / 0.000 / 0.000 | 4.38 | 1.94 |
| t26, 90 | 0.003 / 0.004 / 0.003 / 0.013 | 2.40 | 1.44 |
| t26, 140 | 0.005 / 0.002 / 0.003 / 0.005 | 0.79 | 0.16 |
| t392, 40 | 0.000 / 0.000 / 0.000 / 0.000 | 4.92 | 1.68 |
| t392, 90 | 0.001 / 0.005 / 0.006 / 0.008 | 1.27 | 0.17 |
| t392, 140 | 0.001 / 0.001 / 0.011 / 0.007 | 0.76 | 0.31 |
| t392, 200 | 0.003 / 0.004 / 0.002 / 0.004 | 0.77 | 0.79 |

So the two scales behave differently. Predicting the garment's configuration four
seconds ahead is hopeless; predicting where the sleeve opening sits relative to the
arm, and how much of the arm is covered, is reproducible to a few millimetres and to
about 0.01 of coverage. Task-level labels collected from restored states are
therefore usable, while state-level ones are not.

The third row of each block also shows where the divergence comes from: once the
sleeve is on the arm (`elbow_hook`, `align_pitch`), contact constrains the garment
and the growth rate falls by one to two orders of magnitude against the free-hanging
`middle` stage. Contact suppresses divergence here; free fabric amplifies it.

## What this changes

1. **Any counterfactual comparison needs repeats, and now we know how many.** A
   difference of less than about 0.01–0.03 in sustained coverage over a 40-decision
   window is inside the noise of one command sequence with itself. Two macros whose
   returns differ by more than that are separable with a handful of repeats; below
   it, no number of repeats of a single pair will settle the ranking cheaply.
2. **Long differentiable horizons are not a matter of engineering.** The measured
   amplification is what the literature predicts for this class of system: Metz et
   al. 2021 ([2111.05803](https://arxiv.org/abs/2111.05803)) show the product of
   per-step Jacobians, and hence the reparameterization gradient, diverges
   exponentially when the system is chaotic and name contact simulation as such a
   system, with gradient variance growing exponentially in unroll length even in
   double precision; Suh et al. 2022 ([2202.00817](https://arxiv.org/abs/2202.00817))
   show that in a chaotic system the first-order gradient estimator's variance
   eventually exceeds the zeroth-order one's as the horizon grows. Both were verified
   against the primary sources.
3. **Simulation-to-real comparison should be distributional.** Two runs of this
   simulator from the same state with the same commands already differ by centimetres
   of garment configuration, so agreement with a real trajectory at that level is not
   a meaningful target; agreement of task-level quantities and their spread is.

## Limits

Two garment/body cells, one seed each, five repeats, a 40-decision window, the
scripted expert's commands. The separation of *numerical* non-determinism from
*dynamical* sensitivity is indirect: the perturbation-independence shows that
amplification, not the size of the initial difference, sets the scale, but a
deterministic-kernel or CPU run would be needed to attribute the seed of the
divergence. Closed-loop policy feedback can amplify further than these fixed
sequences: an earlier probe measured a 0.067 coverage range over three closed-loop
continuations against 0.049 for fixed actions
(`2026-09-17-dressing-research-redesign.md`). Nothing here measures real cloth.
