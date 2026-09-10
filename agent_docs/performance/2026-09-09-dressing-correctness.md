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

## Matched speed comparison

Both sides are measured on this machine. Newton figures are the medians of
its own `perf/` scalars from uncontended runs; the IPC figures are the
fastest observed 20-step windows of runs that shared the GPU, so they are
lower bounds on this port's speed. Decision cost is not comparable across
decision rates, so the invariant to compare is simulation steps.

| Configuration | Environments | Simulation steps per decision | Seconds per vector step | Replay transitions per second | Simulation steps per second |
|---|---:|---:|---:|---:|---:|
| Newton, horizon 900, decimation 1 | 40 | 1 | 1.03 | 38.8 | 38.8 |
| Newton, horizon 150, decimation 6 | 40 | 6 | 3.48 | 11.5 | 69.0 |
| This port, horizon 900, repeat 1 | 16 | 1 | 1.56 | 10.3 | 10.3 |
| This port, horizon 150, repeat 6 | 16 | 6 | 9.50 | 1.7 | 10.1 |

This port's physics throughput is the same 10 simulation steps per second at
either decision rate: the solve dominates and the 16 gradient updates per
vector step are small beside it. Newton's rises from 38.8 to 69.0 because its
36 updates per vector step are a fixed cost that six simulation steps
amortise. The gap is therefore 3.8 times at decimation 1 and 6.8 times at
decimation 6, per simulation step and per transition alike.

Budget, not rate, is what the zero results turn on. The Newton run that first
reached the upper arm (single cell, human 6, tshirt_68) did so at 350,000
transitions at decimation 1, which is 350,000 simulation steps. That budget
costs 1.4 to 2.5 hours in Newton and 9.5 hours here, and it is the same 9.5
hours at either decision rate. The current curriculum run stands at 147,840
simulation steps after 9.1 hours of contended wall clock, 42 per cent of it.

## Where the dressing step's time goes

Profiling one simulation step of the dressing environment, with the coupler's
own phases timed individually and three other jobs sharing the GPU, so the
absolute numbers are inflated but the shares are not:

| Phase | 16 environments |
|---|---:|
| `ipc_world.advance`, the libuipc solve | 1848 ms |
| `ipc_world.retrieve` | 3.1 ms |
| Genesis rigid store and every other coupler phase | 0.5 ms |
| Full `scene.step` | 1810 ms |
| Garment position read-back | 0.1 ms |
| Observation construction, once per decision | 1049 ms |

Genesis contributes nothing measurable. The dressing scene has no robot: the
arm is a fixed affine body inside libuipc and the garment is a native libuipc
shell, so Genesis is a scene container and the IPC solve is the whole cost.
The earlier record's "68 ms of a 120 ms scene step" came from the cloth-drag
task, where a Franka arm and its inverse kinematics ran in Genesis; it does
not describe this task. Genesis's own speed comes from its Taichi rigid, PBD,
and MPM solvers, none of which is on this path, so nothing in this port can
inherit it while the requirement is penetration-free IPC contact.

Batched scaling was previously recorded as linear, which is wrong for the
current environment. Measured under the same contention:

| Environments | Milliseconds per simulation step | Per environment | Simulation steps per second |
|---:|---:|---:|---:|
| 1 | 666 | 666 | 1.5 |
| 4 | 1479 | 370 | 2.7 |
| 8 | 1758 | 220 | 4.6 |
| 16 | 1848 | 116 | 8.7 |
| 32 | 2396 | 75 | 13.4 |
| 64 | 5060 | 79 | 12.6 |

The solve does not fill the GPU until roughly 32 copies: going from 16 to 32
raises throughput by 1.54 times at no cost in decisions, and 64 gives nothing
back. The 350,000-simulation-step budget at which the reference first reached
the upper arm therefore costs 11.2 hours at 16 environments and 7.3 hours at
32. Training runs should use 32.

## Online drape bake in libuipc, replacing the offline pre-worn states

The owner asked why the states must be pre-baked at all: `ppf-contact-solver`
baked them offline only because Newton's own VBD cloth collapses a closed
tshirt's cuff opening, and libuipc is a solver of the same class. The bake is
now `uipc_manip.dressing_bake`: it takes the raw garment mesh, applies Newton's
canonical pre-transform, pins the grasp patch together with the six
opening-polygon vertices, draws that pin set open along the garment's pull axis,
and lets gravity drain the fabric, all in a standalone one-garment libuipc
world. Semantics are then measured off the result and reproduce the offline
socket frame to 2.6e-8. `uipc_manip.dressing_live` places the drape on any
cached body through Newton's `runtime_align` map.

Two consequences. The garment spawns a clearance step *outside* the fingertip,
so it cannot interpenetrate the arm by construction: no arm erosion, and none
of the bake's six rejected cells. And the garment axis is free, so the 23-cell
pre-worn cache becomes 40 cells, five garments on eight bodies.

The material had to be found rather than inherited, and one finding is a defect
in the current episode cloth. libuipc's strain-limiting shell measures stretch
as `E*2r/(1-nu^2)` but shear as `E/(2(1+nu))`, with no thickness factor, so
sharing one modulus between them (the single-argument overload this port uses)
makes shear about `1/(2r)`, here 3,300, times stiffer than stretch and the sheet
effectively unshearable. Baking tshirt_26 and measuring how far the free fabric
sags against the offline drape's 0.112 m:

| Stretch modulus | Shear ratio | Bending | Free fabric sag | Opening radius |
|---:|---:|---:|---:|---:|
| 6e4 (episode setting) | shared | 10 | 0.007 m | 9.93 cm |
| 6e4 | 1/100 | 10 | 0.014 m | 9.93 cm |
| 6e4 | 1/100 | 0.1 | 0.037 m | 9.92 cm |
| 6e3 | 1/100 | 0.1 | **0.105 m** | 9.92 cm |
| 6e2 | 1/100 | 0.1 | 0.216 m | 9.93 cm |
| 6e1 | 1/100 | 0.1 | 0.540 m | 9.93 cm |

A 6 kPa stretch modulus with a hundredth of that in shear and bending 0.1
reproduces the reference drape (0.105 m against 0.112 m) while the opening holds
at 9.92 cm against 9.91 cm. Those are the bake defaults. `DressingConfig`'s
episode cloth is ten times stiffer in stretch, a hundred times in bending, and
about 3,300 times in shear; that is recorded, not yet changed, and is a
candidate explanation for a sleeve that does not deform around the hand.

Each bake runs in its own interpreter and is cached by a content hash of its
inputs. That is not a convenience: libuipc's sanity checker carries state
across worlds in one process, so a second garment's repair round sees the first
garment's checks and fails without a report of its own. The offline tool ran one
container per garment for the same reason.

The raw meshes carry illegal primitives in the canonical rest pose, of two
kinds that need opposite repairs. Parts that merely pass within the summed
collision radius need a thinner radius, not surgery, and the bake shrinks it by
a quarter per round down to 1 um; this is what unblocked tshirt_392, whose rest
mesh has two edges 19 um apart against a 300 um summed thickness. Triangles that
genuinely cross have to be separated: dropping them, which is what the offline
bake did, leaves orphan vertices that libuipc's volume check then rejects
without a report, so the bake instead displaces the reported vertices along
their own normals, doubling the step each round.

| Garment | Vertices | Source | Opening radius | Free-fabric sag | Bake |
|---|---:|---|---:|---:|---:|
| tshirt_26 | 3,889 | online | 9.92 cm (offline 9.91) | 0.105 m (offline 0.112) | 46 s |
| tshirt_392 | 6,837 | online | 8.09 cm (offline 8.05) | 0.076 m (offline 0.187) | 86 s |
| tshirt_4 | 5,761 | offline | 9.11 cm | | |
| tshirt_68 | 3,529 | offline | 8.27 cm | | |
| hospital_gown | 10,436 | offline | 8.14 cm | | |

Two garments bake online today. The other three cross triangles that neither
separation over six doubling rounds nor face dropping repairs, so they keep
Newton's offline drape, which is the geometry the previous pipeline used; the
factory falls back to it automatically and records which source each cell came
from. All five compose with all eight bodies, so the cell count is 40 either
way, and every placed garment clears the arm by 1.7 to 5.3 mm.

## The body is generated too, so no saved state is read

`uipc_manip.dressing_body` samples a body the way Wang's `gen_human_mesh.py`
does: ten SMPL-X shape coefficients from U(-2, 5), a seated pose with the left
arm tucked and the right shoulder and elbow randomised, the right hand pinned to
a relaxed fist, and a standing height from U(1.5, 1.9). The right-arm collider
is not a stored index list: it is every vertex whose dominant SMPL-X skinning
weight lies on the right collar-to-fingertip chain, so it follows the body
through every shape and pose sample. The dressing landmarks use the reference's
own construction, the midpoint of each section's upper and lower surface vertex
with the joint-derived hand offset carried onto the finger.

Generated arms match the cached ones: forearm 33.8 to 40.7 cm against 37.0,
upper arm 25.8 to 31.6 cm against 28.2, 1,409 arm vertices against 1,307. A
body is now a seed, so the cell count is unbounded rather than 23. The `smplx`
package is pure Python and was copied into the Genesis environment from the
Newton checkout; the model files stay where Newton keeps them.

## The episode cloth is far stiffer than the reference, and it costs upper-arm progress

The shear defect above was measured on the drape. Its effect on the task was
then measured directly: the same scripted expert, the same cell, 150 decisions
of six simulation steps, the cuff held at 1e6, changing only the cloth.

| Cloth | Forearm ratio | Upper-arm ratio at step 150 | Largest held-vertex error |
|---|---:|---:|---:|
| Episode setting: 6e4 stretch, shear shared, bending 10 | 1.00 by step 120 | 0.040 | 5.8 mm |
| Reference-matched: 6e3 stretch, shear 1/100, bending 0.1 | 1.00 by step 120 | **0.127** | 2.0 mm |

Both dress the forearm at the same rate, so the difference is entirely in what
happens once the sleeve has to deform around the elbow: the corrected cloth
gets three times as far up the upper arm in the same budget, and the expert is a
stage further along its sequence when the horizon ends. Neither reaches the 0.7
threshold in 150 decisions, and this is one cell of one garment, so it does not
by itself explain the zero success rate. It does show that the material this
port inherited suppresses the deformation the task is made of.

## The corrected cloth learns much faster and solves much slower

A controlled run followed: the curriculum teacher at horizon 150 with six
simulation steps per decision, 16 environments, the cuff held at 1e4, the
garment curriculum admitting tshirt_392 at step 600, everything identical to
the recorded baseline except the three cloth settings.

| | Baseline cloth | Reference-matched cloth |
|---|---|---|
| Stretch modulus, shear ratio, bending | 6e4, shared, 10 | 6e3, 1/100, 0.1 |
| Forearm ratio at 4,808 replay transitions | 0.000 | **0.667** |
| Upper-arm ratio there | 0.000 | 0.008 |
| Return there | -14.7 | +10.2 |
| Forearm ratio at 14,408 transitions | 0.344 | 0.637 (return +17.5) |
| Forearm ratio at 24,648 transitions | 0.441 | not reached yet |

At equal data the corrected cloth is not marginally better: it passes in one
evaluation what the baseline had not reached in three, and it is the first run
of any kind here to show a non-zero upper-arm ratio from a learned policy.
Its later evaluations are the first dressing successes of this port:

| Vector step | Replay transitions | Success rate | Final upper-arm ratio | Final forearm ratio | Return |
|---:|---:|---:|---:|---:|---:|
| 600 | 4,808 | 0.00 | 0.008 | 0.667 | +10.2 |
| 1,200 | 14,408 | 0.00 | 0.000 | 0.637 | +17.5 |
| **1,800** | **24,008** | **0.50** | **0.376** | 0.789 | **+69.0** |
| 2,400 | 33,608 | 0.00 | 0.227 | 0.500 | +32.2 |

At 24,008 transitions eight of sixteen deterministic episodes ended with the
upper arm dressed past the 0.7 threshold, all of them tshirt_26; tshirt_392
scored zero. No simulation errors, the held cuff within 1 cm throughout. The
baseline cloth at the same step had 0.00 success and a 0.441 forearm ratio.
The reference's own best across 33 evaluated runs is one success in 40
episodes after 2.98M transitions.

The next evaluation reported zero success with the upper arm at 0.227. A
delegated diagnosis established that this is not a collapse and that the
"0.50" needs restating.

Every slot of a garment holds the same cell, `reset` restores one fixed
snapshot, and the evaluation used one fixed seed block, so the eight rollouts
per garment differ only in observation noise: the effective sample size is two,
and the pooled success rate can only be 0, 0.5 or 1. The peak is tshirt_26
finishing at an upper-arm ratio of 0.7528 against a 0.70 threshold, a margin of
0.053; the "collapse" is the same garment at 0.4541. One continuous quantity
moved 0.30. tshirt_392 never exceeded 0.065 in any evaluation and its reward
never exceeded 0.283 in 14,408 replay transitions, so the run's ceiling was
0.5 by construction.

The training side moved the other way across that window. Rolling twenty-episode
success went 0.183, 0.253, 0.310 and the three highest-return episodes of the
run all came after the peak; the step-2,400 dip is one synchronised sixteen-slot
episode inside a twenty-episode mean, since all slots share an episode counter.
No quantity in the checkpoints moves discontinuously there: the temperature
decays on a straight line, weight norms rise smoothly, and the critic's bias
against the Monte-Carlo soft return crosses zero between 1,800 and 2,400
(-14.2 to +5.7), which is a real transition but a smooth one.

So the claim this run supports is narrower than "the task is learned": a
learned policy dressed one garment on one body past the threshold once, on an
evaluation too coarse to resolve it, while never learning the second garment.
It does establish that the upper arm is reachable by a learned policy on this
solver, which every earlier result left open.

Three consequences were applied. Checkpoint selection now ranks the continuous
final upper-arm ratio ahead of the thresholded rate, so `best.pt` is no longer
chosen on a 0.05 margin. Evaluation draws a fresh seed block per round. The
training log now carries the temperature: the actor and temperature statistics
report only every fourth update and the per-step budget is a multiple of four,
so the last update of every step was never an actor update and those columns
were empty for the entire run.

Three findings were recorded and not yet acted on. The policy stalls at an
upper-arm ratio of 0.55 to 0.61 while the threshold is 0.70, and the reward is
steepest through the elbow rather than flat, so the reward is not what stops
it. Leaving the arm costs a single-step reward drop of about 2.1, realised nine
times in the replay. And the critic assigns 17.0 plus or minus 2.1 to every
tshirt_392 state against 86.3 plus or minus 24.5 for tshirt_26, with an
action-advantage-to-fitting-noise ratio of 0.20 against 1.15, so nearly half of
every batch contributes policy gradient that is mostly critic noise.

It is also faster, not slower. A first reading of these logs reported a
five-fold slowdown; that was a mistake. The correctness pass added
`simulated_transitions` and `simulated_steps` columns to the training log, and
the baseline run predates them, so reading the fourth column as elapsed seconds
gave the new run a constant 96, which is exactly its 16 environments times six
simulation steps. Against the real elapsed column:

| | Baseline cloth | Reference-matched cloth |
|---|---:|---:|
| Median seconds per vector step | 14.2 | **4.8** |
| Range | 9.5 to 217.7 | 4.4 to 31.6 |
| Wall clock to 8,648 replay transitions | about 5 h | 1.4 h |

Profiling one simulation step directly agrees: at 16 environments, after the
expert has driven the sleeve into contact, the corrected cloth takes 825 ms
against the baseline's 1131 ms. A stiff sheet is not cheap here. It resists the
strain-limiting projection, so the Newton solve works harder, and the baseline's
long tail (218 s against 32 s) is where it fights hardest.

Which of the three changes matters, profiled at eight environments per
simulation step: bending 10 to 0.1 alone takes 1414 ms to 515 ms, the shear
ratio alone 1246 ms, the stretch modulus alone 1251 ms, and all three together
753 ms. Bending carries most of the speed; shear is the one the constitution's
own convention says was wrong.

## What runs beside IPC in this scene, and what could

Genesis registers eight solvers and activates only the ones that own an
entity. Printing them for a built dressing scene:

| | |
|---|---|
| Registered | Tool, Rigid, Kinematic, MPM, SPH, PBD, FEM, SF |
| Active | Rigid, holding one entity: the ground plane, spawned `coup_type="ipc_only"` |
| Coupler | IPC |

So nothing else is running. The arm is a native libuipc affine body and the
garment a native libuipc shell, both created through the coupler's handles, so
the only Genesis solver with work to do owns a ground plane that is itself
handed to IPC. That is why the step profile puts 100 per cent of the time in
`ipc_world.advance`.

What the coupler can compose with is narrower than the solver list suggests.
`IPCCoupler._add_objects_to_ipc` reads the FEM solver and the rigid solver and
nothing else; the file contains no reference to PBD, SPH, or MPM. Volumetric
FEM entities enter as stable Neo-Hookean bodies and rigid links as affine
bodies, so a scene can mix rigid articulations, volumetric FEM, and cloth in
one IPC contact framework, which is what the shipped examples do
(`ipc_robot_grasp_cube`, `ipc_robot_cloth_teleop`, `ipc_objects_falling`). All
of them use `materials.FEM.*`; none uses `materials.PBD.*`.

Cloth in particular has only one path. `FEMSolver` line 1226 says its cloth
entry point adds "vertices and surfaces for rendering only (no physics
computation). Cloth is simulated by IPC". `materials.PBD.Cloth` exists but
belongs to the PBD solver, which the IPC coupler never reads, so a PBD garment
would step outside the contact framework and would not see the arm at all.
There is no mode in which PBD carries the cloth while IPC guards the contact.

The practical consequence for this task: adding a robot arm, a table, or
volumetric objects is supported and would be solved together with the garment,
at the cost of a larger IPC system. Making the cloth cheaper by moving it to
PBD is not, and would in any case give up the penetration guarantee that is the
reason for using this solver.

## Why one world saturates, and what parallelism buys (agent measurement)

A delegated measurement pass on the pure-pyuipc garment world (soft cloth,
copies alternating tshirt_26 and tshirt_392, `Engine.frame_stats()` for
iteration counts, 20 timed steps, GPU shared with two other jobs so absolute
milliseconds are inflated and the foreign load varied two-fold within
minutes; ratios inside one window are what to trust):

| Copies | Degrees of freedom | ms per step | Newton iterations per step | PCG iterations per step | ms per Newton iteration | Copy-steps per second |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.7k | 421 | 3.2 | 302 | 131 | 2.4 |
| 4 | 64k | 710 | 4.0 | 667 | 177 | 5.6 |
| 8 | 129k | 593 | 4.65 | 800 | 127 | 13.5 |
| 16 | 258k | 1148 | 5.05 | 840 | 227 | 13.9 |
| 32 | 515k | 2695 (median 2339) | 6.15 | 1025 | 438 | 11.9 (13.7) |

Two mechanisms, both measured. Below about 16 copies the device is
latency-bound: a fused-PCG iteration is a serial chain of about eight small
kernels plus the multilevel preconditioner on one stream, costing about a
millisecond whatever the size, so the cost per Newton iteration rises only
1.7 times for 22 times the degrees of freedom. By 32 the device is full and
the same ratio is 1.9 for a doubling. On top of that the whole world iterates
for its worst copy: Newton iterations per step rise from 3.2 to 6.15, because
convergence is a global maximum displacement, the CCD time of impact is
global, and every garment sits in one preconditioned system with one
tolerance. Eight identical copies need 3.35 Newton iterations per step; eight
heterogeneous ones need 4.65. Every extra iteration re-runs collision
detection, assembly, the linear solve and the line search for all copies,
converged or not. Host overhead is not a factor: `advance` blocks, `sync`
costs 2.6 ms of 420, and Python runs only the animator callback (11 ms of
1720). The step-time tree at 16 copies and five Newton iterations: linear
solve 0.80 s (PCG proper 0.50), line search 0.38, contact gradient and
Hessian 0.25, broad phase 0.24, total 1.72.

The earlier record's 1.54 times for 16 to 32 copies was the stiff cloth in a
different window and is not confirmed on the soft cloth here (13.9 against
11.9 to 13.7); the mechanism predicts little past 24 to 32 and the pair needs
re-measuring on a quiet GPU before 32 is adopted for this cloth.

libuipc has no stream or asynchronous API: the backend runs on the legacy
default stream, `advance` is synchronous, and `World.sync` is a device-wide
synchronise. Two engines in one process work, but two 8-copy worlds on two
threads run slower than one 16-copy world (10.7 against 13.9 copy-steps per
second) because both threads feed one in-order stream, and two worlds stepped
sequentially gain nothing (14.5).

Across processes the picture matches Newton's own record. Without MPS two
8-copy processes time-slice, each at half speed, for no aggregate gain. An
MPS control daemon and server already run under this user (private pipe
`~/.mps_pipe`, started 2026-08-26); neither training job uses it. Pointing two
probe processes at it with `CUDA_MPS_PIPE_DIRECTORY` gave 16.1 copy-steps per
second against 12.9 without, and 11.8 against 8.9 in a heavier window: 1.25
to 1.32 times, with one pair in a rising-load window showing nothing. MPS
clients share one fault domain, and mixing an MPS process with a non-MPS one
only time-slices.

What this leaves: one 16-copy world per process, optionally two such
processes under the existing MPS server for about 1.25 times; not threads, not
sequential worlds. Beyond that the evidence points at two backend changes,
neither measured: per-world CUDA streams, and letting a converged subscene
stop iterating so it no longer pays for the worst copy's Newton iterations.
The second changes the numerics and would need its own validation.

## The non-physics quarter of a decision (agent measurement, then applied)

A delegated profile of everything outside the solve, 16 environments, six
simulation steps per decision, every timing bracketed by a device
synchronise, on the shared GPU so absolute numbers are inflated and ratios are
what to trust. Of a 6.73 s decision, 4.95 s was physics and 1.78 s the rest:
gradient updates 14 per cent, observation 7.6, reward 4.7.

The observation was not slow because of arithmetic. Building 16 clouds one
environment at a time issued about 3,400 kernels for 6.8 ms of kernel time and
444 stream synchronisations; the time was spent waiting on the host-device
round trips, 32 of them in `unique(dim=0)` alone. The reward's 315 ms was
almost entirely one line: the cuff-to-body distance as a brute-force pairwise
scan over the 10,475-point body, per environment, in float64. One SAC update
spent about half its kernel time in float32 attention, and the host side ran
114 to 218 `.item()` reads per update, all from the default Adam reading two
scalars per parameter tensor.

Three exact changes were made and measured in place at 16 environments:

| Change | Before | After |
|---|---:|---:|
| Observation: one padded batch, one z-buffer per camera, one packed-key `unique`, same per-environment random draws in the same order | 935 ms | 71 ms |
| Reward: KD-tree over the static body cloud, built once per body | about 315 ms | 24.7 ms (all sixteen rewards) |
| Adam with the fused kernel on CUDA, no per-parameter host reads | 137 ms per update under contention | measured 0.88 of that by the agent |

The batched builder is a twin of the per-environment one: a CPU test runs
both on the same clouds with the same seeds, with and without augmentation,
and requires identical outputs to 1e-5 and identical generator states
afterwards. The KD-tree distance is checked against the pairwise scan to 1e-12.

Not applied, with the evidence: bf16 autocast over the encoders halves the
update (0.47 with fused Adam) but is a training-quality change that needs its
own validation run; batch 256 with four updates per step is a protocol change;
`torch.compile` and TF32 gave nothing outside noise; CUDA graphs over the
update are the next lever once it is launch-bound, unmeasured.

Taichi, assessed concretely. libuipc does expose a device-side position copy
(`FiniteElementStateAccessorFeature.copy_position_to` into a torch buffer),
and Genesis's `quadrants` kernels take torch CUDA tensors zero-copy, so the
data path exists. It buys nothing: the host view already costs 0.09 ms because
`retrieve` has copied, and the device-side observation was no faster than the
host one. A Taichi z-buffer matched the torch one at the same speed; a Taichi
dense-grid voxeliser was the one real win, 0.4 to 4.7 ms against 20 to 51 for
`unique`, about 35 ms per decision, half a per cent, for a second GPU runtime
and JIT in the package. Not adopted.

## Solver settings that trade accuracy for speed (agent sweep, then applied)

A third delegated pass swept libuipc's scene configuration on the dressing
environment. Its short probes (30 decisions) never reach contact and only
measure speed on a contended GPU, so they are noise; the decisive series is
100 decisions with the episode cloth at two copies, where the scripted expert
reaches a real forearm ratio and the grasp error is a physics check:

| Setting | ms per simulation step | Forearm ratio reached | Largest held-vertex error |
|---|---:|---:|---:|
| Library defaults, two runs | 442, 502 | 0.873, 0.866 | 40.1, 42.8 mm |
| `use_cuda_graph` 2 | 368 | 0.865 | 43.3 mm |
| `tol_rate` 1e-2 | 330 | 0.859 | 43.8 mm |
| `check_interval` 25 with `tol_rate` 1e-2 | 267 | 0.858 | 43.9 mm |
| **`use_cuda_graph` 2 with `tol_rate` 1e-2** | **247** | **0.862** | **41.8 mm** |
| `newton.velocity_tol` 0.5 | 278 | 0.795 | 60.7 mm |
| All of the above together | 142 | 0.651 | 94.8 mm |

Both adopted settings are now the environment's defaults, together 1.9 times
faster with the reach and the grasp inside the spread of two baseline runs.
`use_cuda_graph` 2 is free of accuracy cost by construction: the fused PCG's
header states that the graph path launches the same kernels with the same
arguments in the same order, and mode 2 runs the whole solve as one
device-side conditional graph with no host round trip inside the loop, falling
back to the block-replay mode where the driver lacks support. `tol_rate` 1e-2
loosens the conjugate-gradient relative tolerance, which is a real numerical
change; it is adopted on the evidence above and should be revisited if a run
shows grasp error drifting.

`newton.velocity_tol` is the knob to avoid. It is the fastest single change
and it is the one that breaks the task: the expert loses a seventh of its
reach and the held cuff lags half again as far. Contact-side knobs did
nothing: a linear-BVH broad phase, a halved `d_hat`, disabled friction, and
the semi-implicit Newton start all landed inside the contention noise.

The sweep ended early on a rate limit, so it produced no written report; these
figures are read from the 39 result files it left in the scratchpad.

## Where the speed comparison stands after this pass

Re-measured with every change of this record in place (reference-matched
cloth, the CUDA-graph and tolerance settings, the batched observation and
KD-tree reward), one decision of six simulation steps, GPU still shared with
the foreign Newton job and this port's own training run:

| Configuration | Environments | Simulation steps per decision | Seconds per decision | Replay transitions per second | Simulation steps per second |
|---|---:|---:|---:|---:|---:|
| Newton, decimation 1 | 40 | 1 | 1.03 | 38.8 | 38.8 |
| Newton, decimation 6 | 40 | 6 | 3.48 | 11.5 | 69.0 |
| This port, 16 copies | 16 | 6 | 2.36 | 6.8 | 40.7 |
| This port, 32 copies | 32 | 6 | 3.06 | 10.5 | 62.8 |

Per simulation step at the matched decimation-6 configuration the gap has
closed from 6.8 times to 1.10 at 32 copies and 1.70 at 16. The earlier figure
was the stiff cloth on the library's solver defaults with the unbatched
observation; component costs at 16 copies are now 323 ms per simulation step
against 1131, observation 75 ms against 935, reward 25 ms against 315. The
32-copy row also settles the earlier open question: with this cloth and these
settings, 32 copies do give more throughput than 16 (62.8 against 40.7
simulation steps per second), which the stiff-cloth measurement had not
confirmed.

Newton still runs more copies per process and reaches a given transition
count sooner: 24,008 transitions, the budget at which this port first dressed
the arm, costs 0.58 hours there against 0.64 at 32 copies here and 0.98 at 16.
What that budget buys differs: here it produced a 0.50 success rate, and the
reference's own 33 evaluated runs peak at one success in 40 episodes after
2.98M transitions. Speed per transition and value per transition are separate
axes and this port is now close on the first and ahead on the second.

## The remaining speed levers, measured

A third pass swept what the earlier ones had left. Its headline is not a solver
knob: on the strict 100-decision expert protocol the corrected cloth wins on
every axis at once against the historical stiffness, 116 to 118 ms per
simulation step against 168 to 188, a forearm ratio of 0.955 to 0.966 against
0.861 to 0.869, and a held-cuff error of **3.3 to 5.4 mm against 41 to 43**. A
ten-times tighter grasp for two thirds of the solve time. It is now the
environment default, together with the CUDA-graph and tolerance settings.

Copy count, from one clean window of device work (wall clock was unusable this
pass: four identical 32-copy runs spanned 1374 to 3734 ms):

| Copies | Total vertices | ms per PCG iteration | Copy-PCG-iterations per second | GPU memory |
|---:|---:|---:|---:|---:|
| 16 | 85,808 | 1.019 | 15,694 | 6.3 GB |
| 24 | 128,712 | 1.197 | 20,069 | 7.3 GB |
| 32 | 171,616 | 1.532 | 20,925 | 8.3 GB |
| 48 | 257,424 | 2.282 | 21,042 | 10.2 GB |
| 64 | 343,232 | 3.091 | 20,707 | 12.2 GB |

The device saturates by 24 copies and is flat to 64 within five per cent; only
16 is measurably below at 75 per cent. Memory is 122 MB per copy on a 4.3 GB
base and never the constraint. Cost per iteration follows total world vertices
regardless of how they are partitioned: sixteen gowns at 166,976 vertices cost
1.788 ms per iteration, thirty-two shirts at 171,616 cost 1.615. Keep 32.

Mesh size buys roughly linearly, 1.94 times the vertices for 1.98 times the
time and 2.96 for 3.12, and more than that at 32 copies where the per-iteration
cost is itself linear in the degrees of freedom rather than latency-bound. A
heterogeneous world costs 1.45 times a homogeneous one at matched vertices with
1.53 times the Newton iterations, confirming the worst-copy effect, though the
garments also sat at different task states so the margin is not clean.

Everything else is flat or harmful. `newton.max_iter` at 8 or 16 is inside the
baseline's own spread and barely lowers the mean iteration count, because
unconverged frames push their work into the next step; at 4 it breaks the task
worse than the velocity tolerance did, a 0.747 forearm ratio and 65 mm of grasp
error. `newton.use_adaptive_tol` is not a lever at all: the wheel refuses it as
a reserved key that must stay zero. `linear_system.check_interval` is never
read on the graph-mode-2 path, and `line_search.max_iter` cannot bind because
the energy test accepts on the first trial in every recorded step. A halved
`contact.d_hat` and both alternate broad phases are within one to six per cent
of the default `info_stackless_bvh`, which is the fastest available.

The pass also found a crash: the environment's default garment list named
`tshirt_68`, whose pre-worn cell self-intersects at bodies 0 and 2 although the
cache advertises it, so a run that accepted the defaults died at construction.
The default is now the subset that builds for every cached body, the failure
names the cells in the world, and the assets module's full inventory is renamed
so it is not mistaken for a default.

## tshirt_392 is not reachable at an affordable hold, and it cost the run half its batch

The diagnosis above found the critic assigning 17.0 plus or minus 2.1 to every
tshirt_392 state for the whole run while tshirt_26 climbed from 35 to 86. The
scripted expert explains it. On the corrected cloth at the default hold of 1e4,
150 decisions of six simulation steps:

| | Result |
|---|---|
| Final forearm ratio | 0.864 |
| Final upper-arm ratio | 0.000 |
| Threaded at some point | yes |
| Largest held-cuff error | 26.8 mm |

The sleeve passes the hand and travels the forearm, and then stops. The grasp
error is the tell: 26.8 mm is past the environment's own 20 mm tolerance, so
the cuff is slipping on this garment, which at 6,837 vertices is nearly twice
tshirt_26. The same expert on tshirt_26 holds 3.3 to 5.4 mm.

The obvious remedy is not affordable. At a hold of 1e6 the same evaluation did
not finish 150 decisions in 90 minutes and was killed; its settle alone
displaces the garment 0.278 m against 0.061 at 1e4. A hold stiff enough to
carry this garment makes the contact solve too slow to train with, which is the
same wall the 1e5 probe hit earlier.

So from the moment the curriculum admitted it at step 600, tshirt_392 occupied
eight of sixteen slots and contributed a policy gradient that the diagnosis
measured at an advantage-to-noise ratio of 0.20 against tshirt_26's 1.15. Half
the batch was noise for the rest of the run, and the run's success ceiling was
0.5 before it started.

The consequence for the protocol: the curriculum order must be measured on the
current settings rather than inherited, and a garment the expert cannot dress
does not belong in the training world until it can. The one-policy record's
pre-flight sweep is where that measurement belongs.

## A device assert that was an initialisation order, not physics

The sixteen-cell live world of tshirt_26 and tshirt_4 on bodies 0 to 7 died at
construction with a CUDA device-side assert, which read like libuipc's
thickness assert and was first blamed on the settle. A delegated pass showed
it is neither. It is Genesis's Quadrants runtime asserting "Out of CUDA
pre-allocated memory" inside `gs.init`, before any libuipc world exists, and it
fires whenever cuBLAS has already started in the process. Generating a body
with SMPL-X on the GPU starts cuBLAS, and the pre-flight script asked the cell
factory for spawn clearances before constructing the environment, so it
generated bodies first:

| Done before `gs.init` in a fresh process | Result |
|---|---|
| nothing, a CUDA tensor, SMPL-X construction, a CPU body | initialises |
| one CUDA matrix product, or a GPU body | asserts |
| Genesis first, then a GPU body | initialises |

A larger Quadrants pool does not cure it. The environment now detects the
order and says so, keeps its cell factory, and exposes `clearances()` so a
pre-flight reads spawn distances after Genesis is up. The settle count was
irrelevant: the crash reproduces identically at five steps.

Constructed in the right order, the world meets the target: 7.4 s to build,
1.32 s per decision over 30 expert decisions at six simulation steps each, a
largest held-vertex error of 3.0 mm, no simulation errors. Subscene isolation
holds: copies overlap in about a 36 cm cube and come within 0.42 mm of each
other without interacting.

The same pass found a separate problem worth fixing: the settle is a free fall.
The drape bake holds the grasp patch and the six opening vertices, while the
episode holds only twelve anchors, so after placement the garment drops 0.17 m
in ten settle steps and 0.47 to 0.88 m in thirty. It never triggers an assert,
but it means the episode does not start from the drape that was baked.

## Live cells: the scripted expert cannot dress them yet

The protocol asks for the expert to be verified on live cells before any
training. Done now, on the committed code (a02eaec5): one scripted-expert
episode of 150 decisions on each of the sixteen cells tshirt_26 and tshirt_4
times SMPL-X bodies 0 to 7, cuff strength 1e4, run twice.

| Anchors | tshirt_26 forearm reached | tshirt_26 final upper arm | tshirt_4 forearm reached | Settle |
|---:|---:|---|---:|---:|
| 12 | 1 of 8 | 0.275 on body 3, 0 elsewhere | 0 of 8 | 0.879 m |
| 48 | 8 of 8 | 0.054 to 0.484 | 0 of 8 | 0.860 m |

No cell reaches the 0.70 threshold under either count, and no simulation
error occurred. With 48 anchors the held-cuff error is 2.7 to 4.2 mm on five
tshirt_26 bodies but 27 to 49 mm on the three with the lowest upper-arm ratio
(bodies 0, 1, 6), so the grip slips late in the pull there. The steps took
6.3 and 7.4 s per decision because four processes shared the GPU; those are not
throughput figures. A cell the expert leaves at zero is an asset defect, so
fifteen of these sixteen cells could not have trained anything at twelve
anchors, and none of the eight tshirt_4 cells can at either count.

A geometry probe on three of the cells finds the defect in the placement. It
measures the opening centre against the forearm, fingertip to elbow, which is
the segment the hand travels through first:

| State | Along forearm from fingertip | Off the forearm axis | Opening radius |
|---|---:|---:|---:|
| Newton cache file, all 23 cells | -8.8 to -9.4 cm | 0.0 cm | 8.1 to 9.9 cm |
| Live cell as placed | -15.7 to -17.2 cm | 10.3 to 12.4 cm | 9.5 to 9.9 cm |
| Live cell after the settle | -13.2 to -30.5 cm | 18.4 to 18.8 cm | 9.8 to 10.3 cm |

The placement puts the opening centre exactly one clearance, 20.0 cm, out along
the fingertip-to-shoulder chord. That chord is 31 to 38 degrees off the forearm
on these bent arms, so the opening starts about one radius off the axis the hand
travels along. The cache file's states are coaxial with the forearm at 9 cm,
with the opening plane facing along it (normal cosine 0.94 to 1.00). The free
fall then drops the opening a further 6 to 13 cm. The expert dresses only where
its approach happens to re-centre the opening. On tshirt_26 body 3 the opening
crossed the fingertip plane 1.8 cm off axis and the forearm filled. On body 0
it crossed 8.3 cm off axis against an 8.7 cm radius and missed. On tshirt_4
body 0 it crossed near the axis, but the opening had turned almost parallel to
the forearm (normal cosine 0.05 to 0.32), and the hand never entered.

So the multi-cell run is not launched on these cells. The next placement puts
the socket on the forearm axis at about Wang's clearance, then moves it out only
as far as the garment needs to start clear of the arm. It also keeps 48 anchors
and settles the drape under the episode's own pin set; the expert must be
re-verified before training.

### Forearm-axis placement and the episode length

The cache file's cells are not physically settled states. The loader reads
their positions unchanged, yet every body shows the same opening-to-grip offset
for a given garment and the opening sits on the forearm axis to within 0.1 cm,
so upstream each garment's one drape was placed rigidly on each arm. They are
also pre-worn: the garment body lies 11 to 15 cm toward the elbow from the
opening, the arm inside the sleeve tube, and they build only because the cache
path erodes the arm collider by 6 mm. Measured on the CPU for tshirt_26 and
tshirt_4 on bodies 0 and 3, relative to the opening centre along the forearm:

| Placement | Garment body | Grip along forearm | Grip above opening | Nearest cloth vertex to arm |
|---|---:|---:|---:|---:|
| Cache file cells | +11 to +15 cm | +9 to +21 cm | +10 to +18 cm | arm eroded 6 mm |
| Chord, 20 cm, turned outward (committed) | -4 to -25 cm | -5 to -11 cm | -2 to +7 cm | 25 to 77 mm |
| Forearm axis, 9 cm, turned outward | -4 to -17 cm | -12 cm | +1 to +8 cm | 1.3 to 8.9 mm |
| Forearm axis, 9 cm, pre-worn | +4 to +17 cm | +12 cm | +10 to +14 cm | 1.7 to 5.9 mm |

The pre-worn forearm placement reproduces the cache geometry the expert was
tuned on, but libuipc refuses it at 9 and 11 cm even with the arm eroded 6 mm,
on tshirt_26 body 3 and tshirt_4 body 0; vertex distances overstate the margin
against its edge-triangle check. Turned outward on the forearm axis, a world of
tshirt_26 bodies 0 and 3 and tshirt_4 body 0 is refused at 9 cm on tshirt_4
alone and builds at 12 cm. There, with 48 anchors and 150 decisions, the
expert reaches the upper arm at 0.71 on tshirt_26 body 0, the first live cell
past the threshold, and 0.51 on body 3; after the settle the opening sits 3.1
to 4.3 cm off the axis instead of 18.4 to 18.8. tshirt_4 still fails: its
opening reaches the fingertip on the axis, then turns almost parallel to the
forearm (normal cosine from 0.71 down to 0.08) and closes from 9.4 to 5.0 cm,
and the hand never enters.

No single clearance builds the sixteen-cell world on the forearm axis. At 12 cm
libuipc refuses tshirt_26 on bodies 2, 4, 5, 6 and 7, at 14 cm on bodies 1, 2,
4, 5 and 7; every tshirt_4 cell builds at both. Moving out does not help
monotonically, so the contact is not the sleeve reaching the hand along the
axis. Each cell needs its own clearance, chosen by an exact check against the
whole arm mesh.

The episode length also caps the expert. On the committed placement with 48
anchors, run for 300 decisions instead of 150:

| tshirt_26 body | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Final upper arm | 0 | 0.118 | 0.989 | 0 | 0 | 0.936 | 0 | 0.952 |
| Highest upper arm | 0.218 | 0.156 | 0.989 | 1.000 | 0 | 0.936 | 0.141 | 0.952 |
| Expert finished at decision | | | 217 | 212 | | 249 | 274 | 222 |
| Largest held-cuff error, mm | 29.3 | 48.7 | 7.8 | 8.1 | 35.0 | 16.3 | 35.1 | 13.5 |

Three bodies succeed, against none at 150 decisions. The expert reaches the
fingertip at decision 40 to 51 and starts its last pull at 133 to 225, so a
150-decision episode ends it mid-pull; the 20 cm approach alone costs about 45
decisions. Body 3 read 1.000 at decision 200 and 0 from decision 225, after the
expert had finished at 212 with the cuff still held. The progress metric casts
its upper-arm ray from the shoulder toward the elbow and keeps only hits in
front of the origin (`line_triangles`, `t >= 0`, as Newton's `_line_triangles`),
and the expert's last target overshoots the shoulder by 10 cm, so an opening
pushed past the shoulder is invisible and both ratios read 0. The expert's
ceiling is therefore its highest reading, not its last: four of eight bodies
reach the upper-arm threshold within 300 decisions (2, 3, 5, 7). The four that
fail are the four whose grip slips by 29 to 49 mm. The episode length for
training is to be set from the expert ceiling on the final placement, which
shortens the approach.

A stronger grip does not rescue them. At a cuff strength of 1e5 instead of 1e4,
the same four bodies peak at 0.232, 0.174, 0 and 0.144 on the upper arm against
0.218, 0.156, 0 and 0.141, and their held-cuff error stays at 16 to 46 mm. A pin
ten times stiffer that leaves the error where it was is being displaced by
contact, not stretched, so the lever is the placement and the strength stays
1e4. The stiffer pin also cost 3.6 s per decision for four cells against 2.4 s
for eight at 1e4, both on a shared GPU.

### Per-cell clearance on the forearm axis

The live factory now builds the socket on the fingertip-to-elbow axis and
chooses each cell's clearance with an exact check against the whole arm mesh:
any edge-triangle crossing in either direction counts as zero gap, otherwise the
true point-to-surface distance. The check agrees with libuipc's build on every
cell tried. The search starts at 9 cm, steps out 1 cm at a time until the garment
is at least 3 mm from the arm, does not assume a larger clearance is safer, and
raises `NoClearPlacement` past 50 cm. The bake re-keys its cache (revision 2):
its crossing repair now moves the offending vertex along the crossed triangle's
normal, and it opens reported close pairs to the full gap instead of shrinking
the collision radius. Every cell of the four garments builds on bodies 0 to 7:

| Garment | Drape | Clearance, bodies 0 to 7 (m) | Surface gap | Settle |
|---|---|---|---:|---:|
| tshirt_26 | online | .09 .16 .16 .09 .16 .18 .13 .16 | 3.1 to 6.3 mm | 0.605 m |
| tshirt_392 | Newton offline | .09 on all | 25.7 to 35.3 mm | 1.545 m |
| tshirt_4 | Newton offline | .11 .12 .12 .12 .12 .13 .12 .12 | 5.0 to 11.2 mm | 0.745 m |
| tshirt_68 | online | .09 .20 .19 .14 .20 .21 .19 .20 | 3.4 to 11.2 mm | 0.673 m |

The expert over 300 decisions with 48 anchors, highest upper-arm reading:

| tshirt_26 body | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Committed placement | 0.218 | 0.156 | 0.989 | 1.000 | 0 | 0.936 | 0.141 | 0.952 |
| Forearm axis | 0.984 | 0.117 | 0.999 | 1.000 | 0.231 | 0.942 | 0.136 | 0.948 |
| Held-cuff error on the forearm axis, mm | 3.2 | 41.6 | 9.4 | 17.3 | 27.6 | 17.3 | 36.0 | 13.3 |

Five of eight tshirt_26 bodies reach the threshold against four. Only body 0
changed, and its held-cuff error fell from 29.3 to 3.2 mm with the same pin,
which confirms that the slip is contact set by the placement. Bodies 1, 4 and 6
fail under both placements at the same clearances as bodies that succeed, so
that failure follows the body. tshirt_4 reaches the forearm on none of eight.
The settle is still not an equilibrium: the placement tilts the baked down
direction by 50 to 177 degrees, the sixteen-cell settle moves a vertex 0.743 m
against 0.879 m before, and tshirt_392 starts upside down under the measured
roll. The anchor default is now 48 and the cuff strength stays 1e4. Held-cuff
errors and rates above were measured on a GPU shared with other jobs.

### The other garments' ceilings, and the first multi-cell training set

Same placement, 48 anchors, 300 decisions, highest readings over the episode:

| Garment | Reaches the forearm | Passes 0.70 on the upper arm | Notes |
|---|---:|---:|---|
| tshirt_26 | 8 of 8 | 5 of 8 | bodies 0, 2, 3, 5, 7 |
| tshirt_68 | 5 of 8 | 3 of 8 | bodies 0, 2, 3; body 5 peaks at 0.541, body 6 at 0.197 |
| tshirt_4 | 0 of 8 | 0 of 8 | the opening closes against the forearm |
| tshirt_392 | 0 of 8 | 0 of 8 | the drape hangs upside down and the expert never leaves its elbow pull |
| hospital_gown | 0 of 8 | 0 of 8 | the held cuff drifts by up to 143 mm |

tshirt_68 bodies 1, 4 and 7 never reach the forearm. All three sit at a 20 cm
clearance, but body 5 threads at 21 cm, so the clearance alone does not explain
them. The first multi-cell run therefore trains tshirt_26 and tshirt_68 on
bodies 0 to 7 with bodies 6 and 7 held out: 28 copies, two slots per training
cell, 300 decisions of six steps, 48 anchors, a 300,000-transition screening
budget, and saved evaluation trajectories so that a tube-coverage metric can
later be checked against the ray metric on the same rollouts. The trainer cannot
drop single cells, so the three tshirt_68 cells the expert cannot thread stay
in: two take 4 of the 24 training slots and one is among the four held-out
cells. Of the held-out cells only tshirt_26 body 7 is dressable by the expert,
which bounds the held-out success rate the scripted expert would score at one
in four; checkpoints are ranked on the continuous held-out upper-arm ratio. The
training log carries the wall-clock seconds spent in action selection, the
environment step and the gradient updates as cumulative `act_s`, `env_s` and
`update_s` columns, evaluation excluded.

### The SAC update was mostly a distance computation

With the phase columns, steps 140 to 200 of the first run took 9.5 s each:
2.0 s in the environment, 7.5 s in the 24 gradient updates and 0.04 s choosing
actions, so one update took about 310 ms. A profile of the update at the run's
exact configuration, on the GPU it shared with that run, put 72% of the
update's GPU time in `torch.cdist`: seven calls per update, one per
set-abstraction level and encoder pass, about 56 ms each, in the path without
the matrix product, which is slow for three-dimensional points. Squared
distances from explicit coordinate differences, compared with the squared
radius, leave every validity mask unchanged over 60 batches at both levels.
Neighbour order differs in 98 of 117 million valid slots through rounding ties,
which the masked max ignores, and a test batch's actor and critic outputs are
identical with gradients within 1.5e-8. Interleaved on the shared GPU:

| Update | ms per update | Speed-up |
|---|---:|---:|
| `cdist` | 616 | 1.00 |
| Explicit squared differences | 210 | 2.93 |
| Plus one ball query per batch for every pass | 187 | 3.30 |

The rest was measured and not applied: moving each set abstraction's first
linear layer before the neighbour gather gave 1.05, evaluating the actor's
feature propagation only at the tool point 0.98, TF32 1.12, bf16 autocast 1.07,
and batches of 128 and 256 cost the same per sample as 64. The reuse is scoped
to one `SACAgent.update`, and its keys carry the tensors' version counters.

Measured in the relaunched run at commit f40fa300, steps 20 to 60:

| Phase per vector step | Before | After |
|---|---:|---:|
| 24 gradient updates | 7.5 s | 2.07 s |
| Environment | 2.0 s | 2.08 s |
| Total | 9.5 s | 4.15 s |

One update now takes 86 ms against 310 ms, and the run spends about as long
learning as simulating. At this rate the 300,000-transition budget takes about
14 hours of stepping plus evaluation.

With the distance computation gone the remaining levers re-measure
differently, interleaved on the GPU shared with the run:

| Update, fixed code | ms per update | Speed-up |
|---|---:|---:|
| fp32 | 177 | 1.00 |
| TF32 matrix products | 158 | 1.12 |
| bf16 autocast over the whole update | 95 | 1.86 |

Per pass on one batch: actor forward 24 ms, target-critic forward 22 ms,
critic forward and backward 54 ms, actor forward and backward 74 ms. On the
update schedule that is 124 ms, of which a critic on a low-dimensional
privileged state would leave the actor's 43 ms, 34%. The whole-update bf16
figure is an upper bound: Q values here run about 86 with a spread of 24, where
bf16 resolves only about 0.5, so a shipped version would autocast the point
encoders alone and keep the heads, targets and losses in fp32. The order that
follows: a privileged-critic variant first, in fp32 and otherwise identical to
the baseline, then encoder-only bf16; overlapping simulation with learning is
worth at most about 15% once the privileged critic lands, and libuipc does
release the GIL in `advance` and `retrieve`, so it stays possible later.

## Differentiable simulation: neither library provides a usable gradient here

The owner asked whether Genesis's differentiability or libuipc's own could train
the policy faster. A delegated pass answered it from source and measurement.

Genesis cannot: its IPC coupler's `couple_grad` is `pass` with the comment "IPC
doesn't support gradients yet" (coupler.py:876-879), and this scene runs no
Genesis solver that has one.

libuipc's diff-sim is an unfinished stub in the 0.0.28 wheel. Any geometry or
contact-model attribute, including the soft constraint's `aim_position`, can be
registered as a parameter through a `diff/<name>` companion attribute
(src/core/core/diff_sim.cpp:37-100), and `broadcast()` writes it; that is all.
The derivative matrices are private with no accessor (diff_sim.cpp:118-121),
the CUDA manager's `update`, `assemble` and `write_scene` are empty and marked
"Waiting for later version merging" (global_diff_sim_manager.cu:187-200), every
reporter registration is commented out (finite_element_method.cu:171-183,
affine_body_dynamics.cu:106), and no constitution implements a parameter
derivative. There is no test or sample that exercises it.

Action gradients are obtainable anyway, without a rebuild: the debug option
`extras/debug/dump_linear_system` writes each Newton iteration's matrix, which
already contains the shell, bending, the soft constraint, contact and friction,
and an implicit-function adjoint through the frames runs in Python on it. Checked
against central finite differences:

| Solver setting | 1 step | 5 steps | 10 steps |
|---|---:|---:|---:|
| Tight, about 6 Newton iterations per step | 2.8 % | 5.9 % | 25 % |
| One Newton iteration, exact linear solve | 0.7 to 0.9 % | 3.5 % | 22 % |
| The environment's settings | 48 % | | cosine 0.1 to 0.4 |

The machinery is exact for one step; the multi-step drift comes from terms the
Hessian leaves out (lagged friction, adaptive contact stiffness) and from its
positive-semidefinite projections. At the environment's own tolerances the
gradient does not predict the simulator: on the dressing cell a gradient step
fails to move the target the predicted way in 3 of 16 trials, a random direction
moves it half as much as the gradient direction, and two identical replays from
the same state already differ by up to 3 mm, which also rules out finite
differences. Gradient-grade accuracy needs a solve eight to twenty-five times
tighter, on a step that is already entirely solve.

Ranked conclusion: keep differentiable simulation out of the SAC loop. It works
for refining the scripted expert's demonstrations over five-step windows with a
tight solve, but the gains are local and the actual failure, reaching the upper
arm, is a strategy problem the gradient does not see. Cloth parameter
identification would need a new CUDA derivative per material model, and the fit
is three scalars already found by a six-point sweep. A SHAC-style analytic policy
gradient is not viable: it needs a batched native adjoint and the slower solve on
every rollout, against a discontinuous reward. Probes and results are in the
session scratchpad (`probe_diffsim.py`, `tiny_ift2.py`, `dress_ift2.py`).

GPU grasp and reachability measurements are recorded below after completion.
