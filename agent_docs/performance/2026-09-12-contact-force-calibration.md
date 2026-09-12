# Contact-gradient unit calibration and its limits

Date: 2026-09-12. Status: unit conversion measured; physical force accuracy not established.

Updated after the [force-learning audit](2026-09-12-force-learning-audit.md): normal equilibrium alone does not validate friction or final-state force readout. The tracked reproducer is `python -m uipc_manip.calibrate_contact_force`.

`uipc.core.ContactSystemFeature` exports the contact energy, gradient and Hessian of a running
scene. The gradient is what a force term in a reward or an observation would read, but it is stated
in no units, so it was never used. This measures the conversion against a known force.

## The measurement

Ten particles of radius 1 cm and density 1000 kg/m^3 fall onto a ground half-plane and settle. At
equilibrium the total normal contact force must equal their weight, so the ratio of the summed
exported gradient to that weight is the conversion. Running the same scene at two time steps
separates a force, whose ratio would not depend on dt, from an incremental-potential gradient, whose
ratio scales with dt squared. The script is `output/uipc_manip/contact_force_calibrate.py`.

| Time step | Total mass | Weight | Summed PH+N gradient, y | Ratio to weight | Ratio / dt^2 |
|---|---:|---:|---:|---:|---:|
| 1/60 s | 0.0418879 kg | 0.410501 N | -1.14028e-4 | 2.77778e-4 | **1.000000** |
| 1/120 s | 0.0418879 kg | 0.410501 N | -2.85070e-5 | 6.94444e-5 | **1.000000** |

2.77778e-4 is (1/60)^2 and 6.94444e-5 is (1/120)^2, so:

> **force in newtons = - contact gradient / dt^2**

exactly, at both time steps. The sign makes the ground push up, as it must.

## What else the measurement establishes

- **Ten primitive types, not five.** `contact_primitive_types()` returns `EE+F, PE+F, PP+F, PT+F,
  EE+N, PE+N, PP+N, PT+N, PH+F, PH+N`: edge-edge, point-edge, point-point, point-triangle and
  point-halfplane, each split into a normal (`+N`) and a friction (`+F`) term. Normal and friction
  forces are therefore separable, which a single contact-impulse figure is not.
- **The export is a sparse doublet list.** `contact_gradient(prim_type, geometry)` fills the
  geometry's instances with `i`, the global vertex index, and `grad`, a 3-vector. A vertex can appear
  more than once and the entries must be summed. `contact_hessian` gives `i`, `j` and the 3x3 block.
- **Internal contacts cancel, as they must.** The eighteen particle-particle entries sum to zero
  while the single point-halfplane entry carries the whole weight. So a force on one body can be read
  by selecting the primitive entries whose vertices belong to it.
- **Friction is reported separately and was zero here**, which is right for a resting stack.

## Why it matters

This supplies model-force units for diagnostics and prospective learning signals. It does not
establish independently measured cloth–skin forces. Successful equilibrium calibration does not
validate cached gradients during motion, friction discretization, or material parameters.

## First measurement in the dressing scene

The scripted expert dressed tshirt_26 on held-out body 14045 for 300 decisions while the contact
gradient was read every fifth decision and split by vertex. The arm occupies the first block of the
global index space and the cloth the second, which the indices confirm: until the sleeve reaches the
arm at about decision 40, every contact entry lies in the cloth block. The readout is `uipc_manip.contact_force.vertex_forces`, selecting the arm's index block.

| Decision | Stage | Upper-arm ratio | Arm vertices in contact | Net normal force | Sum of normal magnitudes | Peak at one vertex |
|---:|---|---:|---:|---:|---:|---:|
| 30 | middle | 0.00 | 0 | 0 N | 0 N | 0 N |
| 45 | middle | 0.00 | 5 | 39.1 N | 39.3 N | 29.9 N |
| 90 | middle | 0.00 | 40 | 45.4 N | 68.2 N | 33.4 N |
| 130 | last | 0.18 | 50 | **321.1 N** | **358.8 N** | **154.6 N** |
| 165 | last | 0.96 | 65 | 107.4 N | 219.6 N | 98.8 N |
| 285 | done | 0.00 | 62 | 32.4 N | 108.2 N | 13.0 N |

- **These are historical exported model-force measurements, not validated human loads.**
  Net arm force, nodal peak and summed nodal magnitudes are different observables. Comparing all
  three to a wrist-force trial stopping threshold is invalid. The previous inference that an
  ordinary sleeve must exert approximately 1 N was unsupported and is withdrawn.
- **Candidate causes remain unresolved:** material and grasp parameters, contact discretization,
  solver tolerance and cached gradient timing. The 321 N sample has not yet been reproduced with
  controlled solver settings and independent force balance.
- **Friction was nonzero in only 13 of 60 historical samples.** A new loaded-particle test
  reproduces zero exported friction under one-iteration termination even while the particle
  moves. Tightening Newton tolerance restores the expected force in that test. This establishes
  a mechanism, not its quantitative contribution to the dressing trajectory.

- **Every number here is one cell, one policy, one run.**

## Limits of the export, from the source

- **Exporter availability must be inspected.** The simplex exporters depend on IPC-specific
  contact systems; an absent channel must not be interpreted as an empty contact set. Inspect
  `contact_export_status`, including normal and friction channel availability independently,
  instead of assuming that every constitution supports every channel.
- **The gradient and the energy describe different configurations.** The gradient is assembled at
  the top of a Newton iteration while the energies are rewritten during line search, so a force read
  after `advance` belongs to the last iterate, not to the frame's final state. At equilibrium the
  difference vanishes, which is why the resting-stack calibration is exact; during a violent contact
  it need not, and that is one candidate for the 321 N peak above.
- **The overloads taking a constitution are dead**: they look an exporter up by a `"#<uid>"` name
  that is never registered, and warn and return nothing.
- **Attribution does not need a guess.** The backend stamps `builtin.global_vertex_offset` on each
  geometry's meta, so a body's own rows can be selected by its own offset and vertex count
  (`contact_force.geometry_vertex_block`) rather than by the order the scene was built in. The arm
  measurement above assumed the arm was the first block, which the contact indices happened to
  confirm.

## Corrected literature interpretation

TaCauchy matches simulated total normal load to measured load before comparing tactile images;
its SSIM does not independently validate force prediction. IsaacIPC evaluates contact-pressure
transfer and provides a rendering bridge, but leaves tangential-traction validation open.
Distributed dressing-force reasoning already exists in Deep Haptic MPC and Visual Haptic
Reasoning. Earlier universal novelty and safety-threshold claims are withdrawn. Primary sources
and training implications are in the [audit](2026-09-12-force-learning-audit.md).

## The normal channel is trustworthy at equilibrium; the friction channel is not usable

Two further scenes, both with an analytic answer.

**Sliding scene.** Particles on a level ground half-plane. The normal force settles at 0.3284 N
against a weight of 0.328401 N, a fourth confirmation. But the first frames read **228.6, 108.0 and
50.0 N** before falling to the weight by frame 5. A transient of nearly three orders of magnitude is
the barrier resolving the initial configuration, and it is read because the gradient is assembled at
the top of a Newton iteration. **That is the most likely explanation of the 321 N peak in the dressing
scene above**, and it means a force read during a violent contact cannot be taken at face value.

**Tilted scene, the decisive one.** Gravity tilted 15 degrees with a friction coefficient of 0.5, so
tan 15 degrees = 0.268 is below the friction angle and the particles must be held at rest by static
friction alone. Equilibrium demands a normal force of 0.317535 N and a friction force of 0.0850832 N.

| Frame | Normal | Friction | Normal / expected | Friction / expected | Tangential drift |
|---:|---:|---:|---:|---:|---:|
| 0 | 154.1 N | 0.0000 N | 485 | 0.00 | 0.0 mm |
| 2 | 1.135 N | **0.0851 N** | 3.57 | **0.9999** | 0.001 mm |
| 3 | 0.2234 N | 0.0791 N | 0.70 | 0.93 | 0.018 mm |
| 4 | 0.0633 N | 0.0000 N | 0.20 | 0.00 | 0.19 mm |
| 25 | 0.3175 N | 0.0000 N | **1.0000** | **0.00** | 1.74 mm |
| 149 | 0.3175 N | 0.0000 N | **1.0000** | **0.00** | 7.27 mm |

- **The normal channel is exact once settled**, ratio 1.0000 from frame 25 on. Three independent
  scenes now agree.
- **The friction channel reported the right answer once, at frame 2, to four digits.** So the
  quantity exists and the units are the same.
- **From frame 4 on it reads exactly zero while friction is demonstrably acting.** The particles
  should be held; instead they creep at a steady 0.045 mm per frame, 7.3 mm over 150 frames, and the
  export says the friction force is zero throughout.
- **So the friction channel cannot be used as a measurement as it stands.** IPC friction is lagged:
  the basis and the normal force it multiplies are fixed at the start of a step, and the friction
  gradient is a function of the tangential slip accumulated *within* the step. Read at the top of a
  Newton iteration, that slip is near zero, so the gradient is near zero — even when the physical
  friction force is not. This matches what the solver's own authors say, that there is no guarantee
  of accurate satisfaction of implicit friction relations, and what an external critique of spurious
  tangential forces in IPC reports.

**Consequence for any shear-based direction.** A distributed skin-traction signal is exactly the
quantity nobody else can produce, and exactly the one this export does not yet deliver. Before it can
be a reward, an observation or a constraint, one of these has to be settled: read the friction
gradient at a point in the solve where the slip is resolved; reconstruct the friction force from the
normal force, the friction coefficient and the slip direction, which are all available; or take the
creep itself as the measurement, since a steady tangential slip under a held contact is the physical
signature of shear.

## Which force readings are reproducible

The dressing episode was run twice from the same seed. The GPU solve is not bitwise repeatable, which
the port already records, so the question is how much that costs a force signal.

| Decision 285, settled | First run | Second run | Difference |
|---|---:|---:|---:|
| Net normal force on the arm | 32.415 N | 30.802 N | 5 % |
| Summed normal magnitudes | 108.169 N | 104.932 N | 3 % |
| Peak at one vertex | 13.033 N | 12.582 N | 3 % |

Against that, the first contact at decision 45 read 39.1 N in one run and 4.5 N in the other, a factor
of nine, with three to five arm vertices in contact either time.

- **A settled force is reproducible to a few per cent; a transient one is not reproducible at all.**
  With only a handful of contacts the total is decided by exactly which vertices touch, and that is
  what diverges first between runs.
- **So a force signal for training has to be a settled or aggregated quantity**, not an instantaneous
  peak: an episode statistic, a per-decision aggregate, or a value read once contact has stopped
  changing. A reward term on the instantaneous peak would be fitting noise.

## A force signal that is already in the loop, and that the robot could feel

The cuff is held by a soft position constraint whose docstring states the equivalence directly: the
"equivalent physical spring stiffness is strength * mass / dt**2" (`dressing_env.py:85-88`). Its
displacement is the tracking error, the largest distance between a held cuff vertex and its commanded
position, computed every sub-step (`dressing_env.py:497-500`), reported in every info, and already
carried in the privileged state as a centimetre-scale field (`dressing_privileged.py:124`).

- **So the grasp force is a free readout**: stiffness times tracking error, needing no contact export.
  Over the episode above the settled tracking error is 3.62 to 3.64 mm and stable to 0.5 %.
- **And it is the one force a Stretch 3 could actually sense**, through joint effort, where the
  contact force on the arm is privileged and can only ever be a training-time signal.
- The physical link is the one that matters for dressing: when the sleeve catches, the gripper has to
  pull harder against the hold, so the tracking error rises. Whether that correlation is strong enough
  to serve as a deployable proxy for the arm's load is not yet measured.

## Solver telemetry cannot gate the transients

The gate was to be a flag: read `Engine.frame_stats()` beside the force and discard the frames the
engine says did not converge. **It does not work.** Ten particles placed touching a ground half-plane,
where the settled answer is 0.410 N:

| Frame | Force | Times the weight | `converged` | Newton iterations |
|---:|---:|---:|---|---:|
| 0 | 1009.4 N | **2459** | **True** | **1** |
| 1 | 482.5 N | 1175 | True | 1 |
| 2 | 228.0 N | 555 | True | 1 |
| 4 | 2.9 N | 7 | True | 4 |
| 5 and after | 0.41 N | 1.00 | True | 1 |

`converged` is true on every frame including the 2,459-fold one, and `hit_newton_limit` and
`hit_line_search_limit` are false throughout. Worse, the **largest spike took the fewest Newton
iterations**, so iteration count ranks the wrong way for the worst case.

**So the gate has to come from the force's own history**, and the readout now records what that needs
rather than guessing a rule: per-vertex contact age, and the magnitude's relative change since the
previous read. In the particle stack that change decays 52, 53, 54, 97 and 86 per cent over the five
transient frames and then steadies below a few per cent.

## The readout, wired in

`DressingConfig.contact_force_readout` turns it on. One export serves every slot —
`contact_force.vertex_forces_multi` splits the ten channels across the arms' index blocks, so a
24-environment world costs ten exports a decision rather than 240. Measured on a four-cell world:

| | Seconds per decision |
|---|---:|
| Without the readout | 0.229 |
| With it | 0.240 |

**Five per cent.** Every decision's info now carries the arm's summed and peak normal force, the same
for friction, the contact count, the contact ages and the relative changes, and separately
`grasp_tracking_m`, the hold's displacement, which is the gripper-side quantity a real robot could
feel. A first run over four cells shows the change measure doing its job: 1.2 and 2.0 on settled
contacts against 83 and 232 where a contact had just formed.
