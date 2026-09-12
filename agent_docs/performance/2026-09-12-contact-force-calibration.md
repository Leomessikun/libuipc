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
