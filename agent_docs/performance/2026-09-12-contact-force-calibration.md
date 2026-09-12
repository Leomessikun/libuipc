# The contact force libuipc exports is exact newtons divided by dt squared

Date: 2026-09-12. Status: measured, reproducible.

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

The dressing reward has no force term and the policy never observes force. Wang's reward uses a FleX
contact-lambda proxy at a weight of 0.001 above a threshold; his group's later work had to learn a
force model from 264 real-robot trials in PyBullet (FCVP, RA-L 2024) because FleX cannot report
forces. Here the force on the arm can be read exactly, per contact type, and separated into normal
and friction, at every step of every environment.

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

- **The forces are far above anything the dressing literature reports as safe.** The CMU line cites
  an 18 N limit estimated from Sawyer joint torques. This expert run peaks at 321 N net on the arm
  and 155 N at a single vertex, and even the settled state at the end holds 108 N summed over 62
  vertices. A t-shirt sleeve resting on an arm should be of order one newton.
- **So either the scene's contact is unphysical, or the expert dresses violently, and nobody has been
  in a position to notice.** FleX cannot report forces at all, which is why the same group learned a
  force model from 264 real trials instead (FCVP, RA-L 2024). This measurement is not a result yet;
  it is the first look at a quantity this project has been producing blindly for months, and the
  candidates to separate are the cloth's stiffness, the 48-anchor soft grip pulling the sleeve into
  the arm, and the barrier stiffness `contact_resistance`.
- **Friction is reported only intermittently**: 13 of 60 samples carry a non-zero friction term, with
  the rest exactly zero, including every sample once the sleeve settles. In the particle calibration
  friction was also zero for a resting stack, which is correct there. Before any shear-based signal
  can be used, this has to be explained: a sticking contact should still carry static friction.
- **Every number here is one cell, one policy, one run.**

## Limits of the export, from the source

- **The barrier-free AL-IPC pipeline exports nothing.** The exporters require
  `SimplexNormalContact`, whose only concrete subclass belongs to the IPC pipeline, so under
  `contact/constitution = "al-ipc"` every call returns no entries rather than failing. A caller
  cannot tell that from a scene with no contact. AL-IPC was already rejected for being 2.2 times
  slower; this closes it for force work as well.
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

## Who else reads forces out of this solver

- **TaCauchy** (IROS 2026, arXiv 2606.20426) is built on UIPC, this codebase. It computes Cauchy
  stress from a hyperelastic law and projects it onto the contact surface to get traction and
  pressure distributions, for vision-based tactile sensors inside Isaac Sim. It reports agreement
  with real tactile responses at SSIM above 0.93 over 1.2556 N to 4.7332 N, and 33.40 FPS for one
  environment against 555 FPS aggregate over 60 [RA].
  - **It does not touch cloth, garments or dressing, and trains no policy.** So it does not take the
    dressing question, and it is the best evidence available that forces from this solver agree with
    a real measurement. The caveat is the regime: a small sensor pad at one to five newtons, not a
    garment over a limb.
- **IsaacIPC** (arXiv 2605.24339) puts GPU IPC into Isaac Sim with a mortar contact potential aimed
  at tactile sensing and contact-pressure resolution [RA]. IPC for robotics is being industrialised,
  which sets a clock on anything that depends on it being unusual.
- **Nobody reads these forces for dressing.** A red-team sweep of 2023 to 2026 found robot-assisted
  dressing to be seventeen arXiv papers in total, every published force figure a scalar at the wrist,
  and an arXiv abstract search for dressing together with shear returning no robot-dressing papers at
  all. Clinical guidance names shear, not pressure alone, as the mechanism of pressure injury.
