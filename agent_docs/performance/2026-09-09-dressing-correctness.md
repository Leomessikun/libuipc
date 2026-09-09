# 2026-09-09 — Dressing correctness and reproducible comparison

## Scope and environment

The owner requested parallel investigation and fixes for the gap between
working Newton dressing policies and this IPC port. The work separates grasp
tracking, training protocol, and trajectory validation. Repository artifacts
remain English; user-facing discussion is in Chinese.

Starting branch: `cloth-cable-manip-rl`, initially clean. Changes are Python
environment/training code; the runtime is the installed `pyuipc` 0.0.28 wheel
in `/home/ge47gax/kun/genesis-world/.venv`, not a build of this checkout.
GPU: RTX PRO 6000 Blackwell Workstation Edition, driver 595.84. An existing
Newton training process and the existing IPC curriculum run share the GPU.
Probe wall times are therefore not uncontended throughput measurements.

## Confirmed implementation problems

- Oversized dressing observations discarded the end of the cloth cloud and
  could discard all cloth when the arm consumed the point budget. Voxel
  centroids have spatial order. Sampling now distributes the budget between
  nonempty segments and selects points uniformly within each. Inputs already
  within budget are unchanged. This is a correctness fix for overflowing
  clouds, not evidence that the historical 768-point runs overflowed.
- Training resume could reconstruct different default environment/discount
  settings while loading saved optimizer state. Resume now restores saved
  configuration and validates compatibility. Rollout collection inherits the
  teacher configuration before constructing its environment too.
- CSV output could omit optimizer metrics that appeared after the first row
  and overwrite previous logs. Logging now preserves its evolving schema and
  previous rows.
- A simulator failure could reset the world and then contribute a transition
  that bootstrapped into the new episode. Training now aborts before inserting
  any transition from a failed vector step.
- Evaluation counts smaller than the vector width sampled only the first
  slots. Evaluations now use complete rounds of all slots and report the
  requested and actual episode counts.
- Curriculum training now counts admitted replay transitions against its
  training budget, and reports simulated transitions and simulation steps
  separately. These quantities are not interchangeable across action repeats.

## Interpretation corrections

The old report compared IPC scripted experts with Newton learned policies on
different cells and budgets, then attributed a similar progress range to a
solver-independent task plateau. That conclusion was unsupported and has
been removed. Zero final success, zero upper-arm coverage, and zero forearm
coverage are distinct outcomes.

The existing `h150x6_curriculum_seed1` run's step-1200 evaluation has mean
final forearm ratio 0.3442168, upper-arm ratio 0, and success 0. Changing both
decision timing and curriculum prevents attributing this improvement to
either change alone. Temperature rescaling corrected the historical Q-value
drift but did not establish improved insertion.

The native soft-constraint implementation minimizes the incremental penalty
`0.5 * strength_ratio * vertex_mass * squared_position_error`. It contains no
`dt^2` factor; converting it to a physical spring stiffness requires division
by `dt^2`. The previous explanation treating the strength as a physical
spring rate and deriving a 0.6-second period omitted that distinction. Runtime
tracking measurements remain empirical evidence independent of that formula.

## Reproduction and validation

Portable package tests run with:

```bash
PYTHONPATH=python /home/ge47gax/kun/genesis-world/.venv/bin/python \
  -m pytest python/uipc_manip/tests -q
```

The diagnostic module records cache/configuration/source fingerprints,
normalized actions, initial and post-action garment states, tool targets,
tracking error, coverage, and blocked translation. It can replay the same
actions and compare traces while checking cell and action scaling. Its CPU
`audit` mode does not establish physics validity or reachability.

GPU grasp and reachability measurements are recorded below after completion.
