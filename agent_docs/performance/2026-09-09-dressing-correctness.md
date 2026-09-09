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
| Forearm ratio at 24,648 transitions | 0.441 | not reached yet |
| Seconds per vector step | 18 to 21 | 96 |

At equal data the corrected cloth is not marginally better: it passes in one
evaluation what the baseline had not reached in three, and it is the first run
of any kind here to show a non-zero upper-arm ratio from a learned policy.

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

GPU grasp and reachability measurements are recorded below after completion.
