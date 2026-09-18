# Does the contact force reproduce? — 2026-09-19

Status: measurement complete. No policy was trained and no solver, reward or
controller setting was changed. It answers whether the simulator's contact force can
carry a training signal, a reward term or a safety rule, which every force-aware
design in this project has assumed without checking.

Protocol: the same one as the
[predictability measurement](2026-09-18-predictability-horizon.md), with the arm's
contact-force readout enabled. The scripted expert drives one slot to a decision, the
world is snapshotted, and the expert's next twenty commands are replayed five times
with bitwise identical commands from the exactly restored state. Forty-two force
channels are recorded per decision beside the task metrics. Cell `tshirt_26/14046`,
snapshots at decisions 90 and 140, 380 decisions in 3.8 minutes, restore error
0.00e+00 m throughout. Command:
`scripts/measure_predictability_horizon.py --force`. Artifacts:
`output/uipc_manip/force_reproducibility_20260919/t26/`.

## No force channel reproduces

Relative spread is the range across the five identical repeats divided by their mean,
averaged over the twenty decisions. At the elbow-hook snapshot:

| Channel | Mean | Relative spread |
|---|---:|---:|
| `vertices_in_contact` | 38.7 | 24 % |
| `upperarm_summed_n` | 37.5 N | 51 % |
| `summed_normal_n` | 49.7 N | 52 % |
| `net_normal_n` | 38.1 N | 56 % |
| `peak_vertex_normal_n` | 14.0 N | 71 % |
| `peak_kpa` | 47.1 kPa | 82 % |
| `settled_summed_n` | 24.8 N | 126 % |
| `summed_friction_n` | 2.1 N | 191 % |
| `settled_elbow_peak_kpa` | 14.7 kPa | 217 % |

Twelve further channels have both mean and spread exactly zero: there is no hand or
forearm contact at this state, so they carry nothing. Of the thirty that do carry a
reading, the steadiest is a count of vertices in contact at 24 %, every summed or peak
normal force is 50 % or worse, and every friction channel is around 190 %. The
"settled" variants, which exist to report only contacts that have persisted and barely
changed, are *less* reproducible than the raw ones, not more.

For scale, the same runs' upper-arm coverage spreads by 0.0065 and 0.0371 — the task
geometry reproduces to a few percent while the force does not reproduce at all.

## What this rules out

A per-decision contact force from this simulator cannot be a policy input, a reward
term, a critic feature or a safety threshold: two runs of the same commands from the
same state disagree by about the size of the reading. Anything built on it would be
fitting the solver's own non-determinism. This is the measurement the project's own
standing rule asks for before a simulator quantity becomes a signal, and the force
fails it, as an earlier informal probe had suggested.

Three uses survive. A contact *count* at 24 % spread is weak but not meaningless.
An average over many repeats of the same state has a standard error that falls as the
repeats grow, so force can be compared between two designs offline, at a cost of a
handful of rollouts per comparison. And a force computed at a converged, quasi-static
state rather than during motion may behave differently; this measurement does not test
that.

## Limits

One cell, two snapshots, five repeats, twenty decisions, one garment. The readout is
the arm-side contact export at the decision boundary; a different sampling point,
a converged-state readout, or a solver tolerance change is untested. Nothing here says
the *physical* forces are unpredictable, only that this readout of them does not
reproduce run to run.
