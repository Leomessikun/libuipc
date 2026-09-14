# Physics gradients as control signals: the solver's own sensitivities for dressing

Date: 2026-09-13. Status: Level 1 probe implemented and run on three cells (seven states, ~45 GPU minutes);
no learner changed, no training launched. This opens a new line, distinct from the memory and
prior lines of the last two days: it uses what nobody else in the dressing literature has —
the simulator's optimizer — rather than treating the simulator as a black box that pays out a
scalar reward.

Tags: [MI] measured here, [RA] read from the source, [E] inferred.

## The idea in one paragraph

An IPC step is not "push the physics forward", it is `x* = argmin_x E(x; u)` with `E` = inertia
+ elastic + contact barrier + friction + the gripper's soft position constraint, whose target is
the command `u`. At the solution `∂E/∂x = 0`, and the Newton solver has just assembled the Hessian
`H = ∂²E/∂x²` to get there. So "how would the cloth have ended up if the gripper had moved a
little differently" is one linear solve with a matrix that already exists:
`∂x*/∂u = −H⁻¹ ∂²E/∂x∂u`, and for the soft constraint `k/2‖x_i − aim_i‖²` the cross term is `−k`
on the held vertices — no new derivative code. From that, `∂coverage/∂u` and
`∂E_contact/∂u` are chain rules: a *direction* per decision instead of a scalar reward. The
scripted expert's known failure — cutting the elbow's inner side, blind to the cloth's resistance
— is precisely a quantity the contact gradient sees and the reward does not. In a regime where we
collect 3% of the reference's samples, a per-sample vector signal is the natural substitute.

## What exists in this tree, checked at the source [MI]

- **A `diff_sim` scaffold, not an implementation.** `include/uipc/diff_sim/` declares
  `AdjointMethodFeature` (`select_dofs`, `receive_dofs`, `compute_dLdP`), `EnableGradFeature`,
  `SparseCOOView` and `ParameterCollection`; `src/backends/cuda/diff_sim/global_diff_sim_manager`
  declares per-frame `H` and `∂G/∂P` triplet buffers, dof offsets per frame, and reporter slots
  (`finite_element_diff_dof_reporter`, `..._diff_parm_reporter`, `codim_1d_..`). But
  `GlobalDiffSimManager::Impl::update/assemble/write_scene` each read "Waiting for later version
  merging", no backend registers the adjoint or enable-grad features, the Python module binds only
  `DiffSim.parameters()`, and no test or example exercises any of it
  (`docs/specification/scene_config.md` is the only mention). Level 2 therefore needs backend
  work; the skeleton it fits into is already there.
- **Contact exports.** `ContactSystemFeature.contact_energy/gradient/hessian(prim_type, geom)` are
  bound in Python; the energy geometry carries `topo` (Vector4i) and `energy` per contact
  primitive, the gradient geometry `i` and `grad` per vertex (`simplex_normal_contact.cu`). The
  environment already turns the gradient into arm forces (`_arm_force_summaries`).
- **Per-frame solver statistics.** `Engine.frame_stats()` returns Newton iterations, linear-solver
  iterations, convergence and line-search data for the last frame.
- **Snapshots.** `World.dump()` writes the current frame's state to disk under the frame number
  (`q.{frame}`, `q_v.{frame}` and the other systems' files) and `World.recover(frame)` restores it
  and sets the engine's frame; the environment's own bookkeeping (anchor, offsets, targets, the
  heuristic's stage) is restored alongside by the probe.
- **The gripper's coupling.** `SoftPositionConstraint` with `constraint_strength`: energy
  `s/2‖x − lerp(x_prev, aim, substep)‖²` per held vertex, the `aim_position` attribute set by the
  animator from the picker's targets each step.

## Where this sits in the literature [RA]

- DiffCloth (Li et al., TOG 2022, arXiv 2106.05306): a differentiable projective-dynamics cloth
  simulator with dry friction; its abstract names "trajectory optimization for assisted dressing,
  closed-loop control" among its applications. So *differentiable cloth for dressing* is not new.
- DiffIPC (Huang et al., SIGGRAPH 2024, "Differentiable solver for time-dependent deformation
  problems with contact"): an analytically derived adjoint through IPC contact and friction, with
  "a small overhead (typically less than 10% for nonlinear problems) over the forward simulation";
  applications are shape, material, friction, initial-condition and trajectory optimisation — not
  policy learning.
- SHAC (Xu et al., ICLR 2022, arXiv 2204.07137): short-horizon actor-critic on differentiable
  simulation — truncated windows against exploding gradients, a smooth learned terminal critic —
  reporting "a greater than 17× reduction in training time over the best-performing established
  RL algorithm" on contact-rich, muscle-actuated locomotion.

The defensible novelty is therefore narrow and specific: **implicit sensitivities of a GPU IPC
solver (barrier contact, friction, sustained cloth–arm contact) used as the actor's gradient
signal in a closed-loop sleeve-threading policy**, with a learned terminal critic for the horizon
the gradient cannot reach. Not "first differentiable dressing".

## Level 1: what the black-box probe measures

`python -m uipc_manip.physics_gradient_probe --garment G --bodies B ... --out DIR` drives the
scripted expert on one held-out cell and dumps three states: *elbow* (the sleeve opening reaches
the elbow, `forearm_ratio ≥ 0.95`), *passed* (upper-arm coverage first ≥ 0.1) and *stall* (no
+0.01 of upper-arm coverage for 15 decisions after the elbow). `World.dump` writes the frame,
`recover(frame)` brings it back with the environment's bookkeeping. From each restored state:

1. **Repeatability** — the same hold command five times: spread of coverage, arm force and contact
   energy, and the largest cloth-position spread. The noise floor.
2. **Locality** — central differences of coverage, the opening's position along the arm, contact
   energy and arm force with respect to the gripper translation at 1, 2 and 5 mm (0.5 mm in the
   first run), three repeats each: cosine similarity between repeats and between step sizes.
   Every ± outcome records the translation the environment executed, because its tether and
   no-move collision rules can shorten or drop a command; a difference per executed metre is kept
   next to the one per commanded metre.
3. **Usefulness** — twelve decisions of 4 mm along the coverage gradient, the axis gradient, minus
   the force gradient, minus the energy gradient, the normalised sum of the coverage and minus-force
   gradients, a release-then-advance sequence (four decisions down the force gradient, eight up the
   coverage gradient), the expert's own commands, three random directions and the hold; each with
   its executed travel, and repeatable (`--walk-repeats`) at states whose decision is noisy. The two
   combined walks were added after the first three cells and run on cell 1's stall state
   (`--states stall`, below).

The cells are the expert's own elbow failures on region 13's held-out poses
(`expert_r13_heldout_s0`): `tshirt_392` on bodies 14045–14049 stall at upper-arm 0.23–0.50 in the
`elbow_hook` stage; `tshirt_26/14049` at 0.12.

## Results

### First cell, the elbow state (`tshirt_392` / body 14046, decision 84) [MI]

The expert reaches the elbow at decision 84 (`forearm_ratio` 0.951, `upperarm_ratio` 0.000, net
normal force on the arm 49.2 N, contact energy 2.46e-5 J-units). World build 14 s; both snapshots
of this first run were this same state (a trigger bug, fixed below), so the second is a repeat.

**Repeatability.** `recover(frame)` is exact: the largest position error after restore is 0.0 m.
The *decision* from the restored state is not: five identical hold commands ended with cloth
positions differing by up to 3.9 mm (2.5 mm in the repeat), net normal force by σ = 2.1 N (3.7 N)
of 49 N, contact energy by σ = 4.3e-7 (2.2e-7), about 1–2 %. The GPU solve is run-to-run
stochastic at the millimetre scale; this is the noise floor, the same fact that closed the
per-decision force line on 2026-09-12.

**Locality.** Central differences of the net normal force with respect to the gripper
translation, three repeats per step size, cosine similarity between repeats:

| step | repeat cosines (run 1) | repeat cosines (run 2) | direction vs 5 mm |
|---|---|---|---|
| 0.5 mm | −0.84, −0.31, 0.55 | 0.31, −0.75, 0.29 | 0.09 / 0.53 |
| 1 mm | −0.06, −0.02, 0.99 | 0.71, 0.98, 0.82 | 0.45 / 0.71 |
| 2 mm | 0.81, 1.00, 0.81 | 0.86, 0.86, 1.00 | 0.96 / 0.97 |
| 5 mm | 1.00, 1.00, 1.00 | 1.00, 0.83, 0.82 | — |

Below 2 mm the difference of two outcomes is the size of the noise (2 mm × 3000 N/m ≈ 6 N against
σ ≈ 2–4 N) and the "gradient" is noise; at 2–5 mm it repeats and the 2 mm and 5 mm directions
agree (cosine 0.96–0.97). The contact-energy gradient behaves the same way (repeat cosines ≈ 1.0
at 5 mm, 2 mm vs 5 mm 0.98). The force gradient at 5 mm is about (−2.4, +2.7, +0.9) kN/m in both
runs: pushing along +x or −y at 1 mm adds about 2–3 N of arm load. **The coverage gradient is
exactly zero at every step size**: Wang's upper-arm ratio is zero until the opening passes the
elbow, so the reward is a plateau across the whole elbow region — the credit-assignment problem
in one number, and the reason a second, continuous progress reading was added to the probe
(the opening's arc length along finger → elbow → shoulder).

**Usefulness.** Eight decisions of 4 mm from the elbow state: no fixed direction — coverage
gradient (zero, so a null walk), minus force gradient, minus energy gradient, three random ones,
hold — produced any upper-arm coverage; the expert's own commands produced 0.015 (run 1) and
0.011 (run 2) while raising the arm load by +32 N (mean 64 N against 29–43 N for the others).
Walking against the force gradient lowered the load the most (−26 N) but moved nothing forward.
At the elbow, then, the force gradient is real and repeatable at ≥ 2 mm, and it is *orthogonal*
to progress: it says where the cloth resists, not where the sleeve advances. Whether the two can
be combined — advance along the axis while descending the resistance — is what the stalled
states (coverage 0.23–0.25, where the coverage gradient exists) have to show.

### Fix and rerun

The stall trigger counted from decision 0, so "stall" fired at the elbow itself. Fixed: the stall
clock starts at the elbow, a `passed` snapshot is taken when coverage first exceeds 0.1, the
continuous `opening_axis_m` reading and its gradient are recorded, the raw ± outcomes are kept
for signal-to-noise ratios, 0.5 mm is dropped from the defaults, walks run 12 decisions.

### Rerun, cell 1 (`tshirt_392` / body 14046): elbow, passed, stall [MI]

Three states of one expert episode (decision period 0.1 s = 6 IPC frames at dt 1/60; per-axis
command cap 8.66 mm; world build 14 s; 3.5–4 min of GPU per state):

| state | decision | upper-arm | arm force | Newton / PCG per frame | position spread over 5 repeats | force spread |
|---|---|---|---|---|---|---|
| elbow (sleeve reaches the elbow) | 85 | 0.000 | 64 N | 3 / 215 | 0.79 mm | 0.68 N |
| passed (coverage first ≥ 0.1) | 126 | 0.102 | 27 N | 1 / 9 | 0.05 mm | 0.04 N |
| stall (no progress for 15 decisions) | 180 | 0.204 | 646 N | 1 / 8 | 0.02 mm | 2.5 N |

**Repeatability is state-dependent.** Restore is exact everywhere (0.0 m). The decision from the
elbow state scatters by ~1 mm and is the only hard frame (3 Newton iterations, 215 PCG); after the
elbow the solve is one Newton iteration and the decision is deterministic to 0.02–0.05 mm. The
millimetre noise that closed the force line is a property of the elbow's bistable contact, not of
the solver in general.

**Locality.** Cosine similarity between three repeats of the finite-difference gradient, and of
the 1 mm and 2 mm directions against the 5 mm one:

| state | quantity | repeat cosine at 1 / 2 / 5 mm | 1 mm, 2 mm vs 5 mm | signal-to-noise at 2 mm (per axis) |
|---|---|---|---|---|
| elbow | coverage | undefined (plateau, exactly 0) | — | — |
| elbow | arm force | 0.25–0.99 / 0.87–1.0 / 0.99–1.0 | −0.75, 0.72 | 6, 8, 3 |
| elbow | contact energy | — / — / — | 0.49, 0.95 | 14, 36, 10 |
| passed | coverage | 1.0 / 1.0 / 1.0 | 0.97, 0.93 | 400, 33, 139 |
| passed | arm force | 1.0 / 0.5–0.65 / 0.97–1.0 | 0.24, 0.17 | 11, 11, 27 |
| stall | coverage | 1.0 / 1.0 / 1.0 | 0.97, 1.00 | 3600, 220, 1400 |
| stall | arm force | 0.99–1.0 / 1.0 / 0.99–1.0 | 0.71, 0.94 | 14, 58, 7 |

After the elbow the coverage gradient is a proper local derivative: it repeats exactly, its
direction is the same at 1, 2 and 5 mm, and its signal is hundreds of times the noise. At the
elbow it does not exist (the reward is flat) and the force landscape is not linear below 5 mm.
The force gradient at the stall is (−4.4, −42, −3.0) kN/m: the sleeve is jammed along y.

**Usefulness — twelve decisions of 4 mm from each state** (Δ axis = advance of the sleeve
opening along the arm; mean F = arm load during the walk; travel = gripper path actually executed
after the tether/no-move rules, 48 mm when every command went through; the expert's own commands
are 8 mm per decision with rotation, so its walks travel 96 mm at the elbow and 86 mm after it):

| direction | elbow: Δ axis / mean F | passed: Δ axis / mean F | stall: Δ axis / mean F / travel |
|---|---|---|---|
| coverage gradient | (undefined) | **+24.1 mm / 16 N** | +4.1 mm / 767 N / **3 mm** |
| axis gradient | **+16.7 mm / 37 N** | +17.6 mm / 27 N | +4.5 mm / 772 N / 3 mm |
| minus force gradient | +13.7 mm / 25 N | +11.5 mm / 18 N | −2.6 mm / 108 N / 48 mm |
| minus energy gradient | 0.0 mm / 40 N | −0.1 mm / 18 N | −9.4 mm / 160 N / 48 mm |
| expert's own commands | +6.2 mm / 63 N | +20.0 mm / **368 N** | +0.3 mm / 621 N / 0 mm |
| hold | 0.0 mm / 34 N | +0.1 mm / 26 N | +2.6 mm / 723 N |
| random (3) | +6.3, +7.0, +11.1 mm | +7.2, +10.1, +14.1 mm | +2.7, +2.7, **+16.0 mm** (194 N) |

Three readings. (i) Where progress is unobstructed the one-step physics direction beats the
scripted expert with half its travel: after the elbow 1.2× its advance and 1.8× its coverage gain
(+0.078 against +0.044) at **one twentieth** of the load — the expert's `elbow_hook` command
(direction (0.42, −0.91, −0.08), into the arm) advances by jamming the sleeve, 27 → 368 N. At the
elbow the axis-gradient walk moved the opening 2.7× as far as the expert at half the load, but
that is the axis proxy: coverage itself rose +0.007 against the expert's +0.028, whose hooking
rotation the fixed translation lacks. The elbow figure is also one draw: the decision there scatters by ~1 mm and each walk is a single
realisation, so the elbow ratios are not established until the walks are repeated (cell 2's elbow
below already differs). (ii) At the stall the first reading, "the tether rejects the pull", was
wrong. The walk rows show what the environment executed: along the coverage and axis gradients
the gripper moved 2.7–3.3 mm in the first decision and then not at all, and all twelve of the
expert's commands were dropped. The tether cannot be the cause — the largest held-vertex gap was
19.6 mm against a 60 mm limit — which leaves the PyFlex no-move collision rule, which drops any
move that would bring the anchor within 12 mm of the arm shell; the probe now records that
clearance to confirm it. So the stall is a lock of the environment's own making: the gripper
stands at the arm and every command with a component into the arm is refused. It also means the
± probes along the forward axes were only partly executed there (at 5 mm the plus side moved
coverage +0.0027 against −0.0038 for minus; at 1 mm the two sides were nearly symmetric), so the
stall's 5 mm coverage gradient is one-sided, its sign right and its magnitude per commanded metre
understated; the probe now records the executed travel of every ± outcome. The two release
directions were executed in full (48 mm): minus the force gradient and minus the energy gradient
took the load from 646 N to 50 N and 28 N at the end (means 108 N and 160 N) and the held-vertex
gap from 19.6 to 7 mm, but retreated 2.6 and 9.4 mm. One random direction, (0.74, 0.54, −0.40),
both released (78 N at the end, mean 194 N) and advanced 16 mm — one of three draws, an existence
result. Its direction is close to the normalised sum of the two gradients the probe already has,
unit(∂coverage) + unit(−∂force) = (0.71, 0.63, 0.31), cosine 0.75; that sum and a
release-then-advance sequence are the two walks added to the probe and not yet run. (iii) So the
one-step sensitivity is the right signal on the approach; at the jam the greedy coverage direction
is refused by the environment and the release direction retreats, and whether a combination of the
two one-step gradients escapes, or a multi-step sensitivity is needed (the adjoint through k frames
with a learned value beyond it, SHAC's split), is the open question the next walks answer. Either
way the environment's rules — the tether and the no-move collision — are non-physical maps from
the commanded to the executed action that no adjoint through the solver sees; Level 3 has to take
its gradient with respect to the executed command or make those rules smooth penalties.

### Cell 2 (`tshirt_392` / body 14049): elbow and stall — the elbow is not one state [MI]

The expert reached the elbow at decision 93, never reached 0.1 of upper-arm coverage, and stalled
at decision 113 (coverage 0.054, in its `align_yaw` stage since decision 99), so this cell has
two states:

| state | decision | upper-arm | arm force | Newton / PCG | position spread over 5 holds | force spread | axis spread |
|---|---|---|---|---|---|---|---|
| elbow | 93 | 0.000 | 85 N | 1 / 21 | **22.9 mm** | 11.2 N | 0.18 mm |
| stall | 113 | 0.054 | 49 N | 3 / 265 | 0.06 mm | 0.12 N | 0.001 mm |

**The elbow state here is a different kind of state from cell 1's.** Five identical hold
commands scatter the cloth by 23 mm and the arm force by 11 N; the hold itself gains 0.033 of
coverage in twelve decisions, the cloth is still settling. The finite differences reflect it:
at 1 mm the repeat cosines are 0.3–0.9 for coverage and negative for the axis, signal-to-noise
about 1; at 2 mm the coverage direction repeats (0.97–0.99, SNR 5–13) but at 5 mm coverage is
flat again and the 2 mm and 5 mm axis directions agree only to 0.63. The walks say the same: the
coverage gradient +0.2 mm, the axis gradient +2.6 mm, the expert +5.4 mm (41 mm executed), the
three random directions +6.7 to +9.9 mm, minus the force gradient +11.4 mm. No one-step gradient
is a usable direction at this elbow state, and cell 1's 2.7× at its elbow was one draw from a
quieter version of the same place.

**The stall here is not a lock.** Every command was executed in full (48 mm), the decision is
deterministic, the gradients repeat exactly at 2 and 5 mm (cosine 1.0), the coverage direction
holds across step sizes (0.85–0.95 against 5 mm) and its signal is hundreds of times the noise.
Walking it works:

| direction | Δ axis | Δ coverage | mean F | executed |
|---|---|---|---|---|
| coverage gradient (0.99, −0.03, 0.13) | **+15.0 mm** | **+0.039** | 42 N | 48 mm |
| axis gradient | +15.0 mm | +0.038 | 42 N | 48 mm |
| minus force gradient | +8.9 mm | +0.029 | 44 N | 48 mm |
| minus energy gradient | +5.9 mm | +0.006 | 30 N | 48 mm |
| expert (`align_yaw`, rotating) | +9.6 mm | +0.007 | 46 N | 96 mm |
| hold | +0.9 mm | +0.007 | 46 N | 0 mm |
| random (3) | +6.8, +7.7, +8.4 mm | +0.016–0.027 | 30–62 N | 48 mm |

The expert stalled because its stage machine was rotating the cuff; the physics said "along the
arm" and twelve decisions of it gave 1.6× the expert's advance and 5× its coverage gain with half
the travel at lower load. This is the case the greedy gradient controller is for.

### Cell 3 (`tshirt_26` / body 14049): elbow and stall [MI]

The expert reached the elbow at decision 93 and stalled at 116 with coverage 0.011 (again in
`align_yaw`, since decision 101). Both states are deterministic (position spread 0.21 mm and
0.12 mm over five holds, force spread 0.2 N) and the solve is easy (1 Newton, 5 PCG).

| state | decision | coverage | arm force | coverage gradient | axis gradient (repeat cos / vs 5 mm / SNR) |
|---|---|---|---|---|---|
| elbow | 93 | 0.000 | 53 N | exactly 0 at every step size | 0.99–1.0 / 0.74, 1.0 / 10²–10³ |
| stall | 116 | 0.011 | 23 N | 1.0 / 0.99, 0.99 / 10³ | 1.0 / 0.96, 0.97 / 10³ |

| direction | elbow: Δ axis / Δ coverage / mean F / executed | stall: Δ axis / Δ coverage / mean F / executed |
|---|---|---|
| coverage gradient | (undefined) | +14.7 mm / **+0.077** / 34 N / 48 mm |
| axis gradient | **+16.6 mm** / +0.004 / 51 N / 48 mm | **+17.1 mm** / +0.070 / 34 N / 48 mm |
| minus force gradient | +8.2 mm / 0.000 / 33 N / 48 mm | +7.1 mm / +0.015 / 18 N / 48 mm |
| minus energy gradient | −3.1 mm / 0.000 / 43 N / 48 mm | −8.5 mm / −0.011 / 17 N / 48 mm |
| expert | +4.6 mm / **+0.020** / 74 N / 35 mm | +11.3 mm / +0.042 / 32 N / 96 mm |
| hold | +0.5 mm / 0.000 / 42 N | −0.4 mm / +0.007 / 27 N |
| random (3) | +7.6, +7.7, +9.0 mm (coverage ≤ +0.017) | +7.7, +8.6, +9.8 mm (coverage ≤ +0.065) |

### Across the three cells: seven states [MI]

| state | position spread (5 holds) | coverage gradient | walk: best physics direction | expert | best of 3 random |
|---|---|---|---|---|---|
| cell 1 elbow @85 | 0.79 mm (3 Newton / 215 PCG) | flat | axis: +16.7 mm, cov +0.007 | +6.2 mm, cov +0.028 | +11.1 mm, cov +0.015 |
| cell 2 elbow @93 | **22.9 mm** (settling) | noise (SNR ≈ 1) | −force: +11.4 mm, cov +0.004 | +5.4 mm, cov +0.053 | +9.9 mm, cov +0.064 |
| cell 3 elbow @93 | 0.21 mm | flat | axis: +16.6 mm, cov +0.004 | +4.6 mm, cov +0.020 | +9.0 mm, cov +0.017 |
| cell 1 passed @126 | 0.05 mm | derivative (cos 1.0, loc 0.93–0.97, SNR 10²–10³) | coverage: **+24.1 mm, cov +0.078**, 16 N | +20.0 mm, cov +0.044, **368 N** | +14.1 mm, cov +0.047 |
| cell 2 stall @113 | 0.06 mm | derivative (cos 1.0, loc 0.76–0.85) | coverage: **+15.0 mm, cov +0.039** | +9.6 mm, cov +0.007 | +8.4 mm, cov +0.027 |
| cell 3 stall @116 | 0.12 mm | derivative (cos 1.0, loc 0.99) | axis: **+17.1 mm, cov +0.070**; coverage: +14.7 mm, cov +0.077 | +11.3 mm, cov +0.042 | +9.8 mm, cov +0.065 |
| cell 1 stall @180 | 0.02 mm | derivative (cos 1.0, loc 0.97–1.0) but one-sided: forward moves refused | none executable forward; −force releases and retreats | 0 mm executed | random_2: +16.0 mm, cov +0.019 (released) |

Four readings across the seven states.

1. **Noise is a property of the state, not of the solver.** Five of the seven states are
   deterministic to 0.2 mm or better over five identical decisions. The two that are not are
   elbow states: cell 1's at 0.8 mm (the one hard frame, 3 Newton iterations) and cell 2's at
   23 mm (cloth still settling; the hold alone gains 0.033 of coverage). Any gradient-based
   learner needs the repeatability gate the force line established, per state.
2. **After the elbow the one-step coverage gradient is a genuine local derivative** in all four
   states: repeat cosine 1.0, the same direction at 1, 2 and 5 mm (cosines 0.76–1.0, mostly
   ≥ 0.93), signal hundreds to thousands of times the noise. **At the elbow it does not exist**
   in any of the three cells, because Wang's upper-arm ratio is exactly zero until the opening
   passes the elbow. That is the same plateau SAC's reward has there; a one-step gradient of the
   task reward cannot be the elbow's signal any more than the reward is.
3. **Where it exists and can be executed, walking it beats the expert and random.** At the three
   post-elbow states without a lock the coverage/axis gradient advanced the sleeve 24.1, 15.0 and
   17.1 mm against the expert's 20.0, 9.6 and 11.3 mm and the best random 14.1, 8.4 and 9.8 mm,
   at half the expert's travel, with more coverage gained (0.078, 0.039, 0.077 against 0.044,
   0.007, 0.042) and lower or equal load (the expert's `elbow_hook` jams at 368 N). At the two
   quiet elbows the axis-gradient walk moved the opening 16.6–16.7 mm along the arm against the
   expert's 4.6–6.2 mm, but coverage followed the expert's hooking, not the translation
   (0.004–0.007 against 0.020–0.028): the axis reading is a continuous proxy, not coverage, and a
   fixed translation without rotation does not hook the cuff. At the noisy elbow no one-step
   direction beats random. At the locked stall the environment refuses the gradient's direction.
4. **Fairness.** The expert's commands are 8 mm per decision with rotation (86–96 mm executed
   after the elbow, 35–41 mm at two elbows where its commands were partly dropped), gradient and
   random walks 48 mm; every elbow walk is one draw. The three "random" directions were drawn
   from one seed per call, so they are the *same* three vectors at all seven states —
   (0.19, −0.20, 0.96), (0.16, −0.82, 0.55) and (0.74, 0.54, −0.40) — and the third, a
   forward-and-up direction, is the best of them at six of the seven; the escape at cell 1's stall
   is therefore "a fixed forward-up direction escapes", not a lucky draw, and "beats the best
   random" is read against three fixed directions. The probe now draws per state. The walks that
   combine both gradients (`combined_gradient`, `release_then_advance`) were added after these runs
   and run once, on cell 1's stall state, next.

GPU used: about 49 minutes of a shared GPU — 7.5 minutes for the first elbow-only run, 34
minutes for the three-cell queue (21:06–21:40) and 4 minutes for the stall re-probe below,
alongside `abl_residual_s1` and another session's evaluation. Everything is in
`output/uipc_manip/physics_gradient_probe/*.json`.

### Cell 1's stall re-probed with the combined walks (`--states stall`) [MI]

The expert was driven again to its stall on the same cell. Because the elbow decision scatters
by ~1 mm the episode diverged a little and stalled at decision 195 instead of 180, at coverage
0.202 and 442 N (before: 0.204, 646 N); the state is the same kind of place. Three things this
run settles:

- **The lock is the no-move collision rule.** The anchor's clearance to the arm shell at the
  stall is **12.6 mm** against the rule's 12 mm; every forward walk executed 0.7 mm and then
  nothing (clearance 12.2 mm), the expert executed 0 of its 141 mm. The ± probes show it per
  axis: at 5 mm the +x side executed 0.83 mm and the −y side 0 mm, so the earlier "central"
  differences there were one-sided; per executed metre the coverage gradient is (1.13, −0.40,
  0.30), per commanded metre (0.57, −0.20, 0.30) — same direction, half the magnitude.
- **Two one-step gradients compose into the escape.** Twelve decisions of 4 mm, all executed:

  | direction | Δ axis | Δ coverage | F at the end / mean | clearance at the end |
  |---|---|---|---|---|
  | coverage gradient alone | +2.1 mm (0.7 mm executed) | +0.006 | 460 / 464 N | 12.2 mm |
  | minus force gradient alone | −3.3 mm | −0.017 | 59 / 68 N | 58 mm |
  | unit(∂coverage) + unit(−∂force), (0.62, 0.66, 0.43) | **+16.5 mm** | +0.029 | **42 / 66 N** | 41 mm |
  | release 4 down the force gradient, then 8 up the coverage gradient | **+17.5 mm** | **+0.053** | 142 / 100 N | 20 mm |
  | expert | +1.2 mm (0 executed) | +0.004 | 391 / 414 N | 12.6 mm |
  | hold | +1.8 mm | +0.006 | 450 / 451 N | 12.6 mm |
  | random, drawn for this state (3) | −2.7, +10.5, +12.0 mm | −0.013, +0.007, +0.011 | 396, 89, 58 N | — |

  Both combinations recover and exceed the earlier fixed forward-up direction (+16 mm, +0.019).
  The sequence is the stronger of the two on coverage: once four release decisions have taken the
  anchor off the wall, the coverage direction computed *at the locked state* executes in full and
  gains +0.053 in eight decisions, more than any walk from any state in the three cells except
  cell 1's passed state.
- **So the jam is not a case for a multi-step sensitivity; it is a case for a policy that
  sequences two one-step ones.** The information needed to escape was in the two gradients the
  probe had at the locked state; what was missing is the decision to release first. That is
  exactly what an actor trained on a short-horizon physics gradient learns, because over h ≥ 4
  decisions `∂coverage_{t+h}/∂u_t` through a release phase is nonzero where the one-step
  gradient's direction is refused.

### The two quiet elbows re-probed with rotation (2026-09-14) [MI]

Same states as the queue (cell 1 @85, cell 3 @93), now with central differences on the two
rotation axes the environment executes (0.5°, 1°, 2.5°; about x is clipped) and every walk run
three times; the tables give mean ± std over the three runs. Twelve decisions of 4 mm and, for
the rotation walks, 2.5° per decision; ~25 GPU minutes for both.

- **Coverage is flat in all five dimensions at both elbows**; the axis reading has a small,
  repeatable rotation gradient (cosine 1.0 at 1° and 2.5°).
- **The decisions scatter by 1–1.5 mm at both elbows, the walk outcomes do not** (std ≤ 0.5 mm of
  advance, ≤ 0.005 of coverage), so the elbow walk figures below are established, not one draw.

| walk | cell 1 elbow: Δ axis / Δ coverage / mean F | cell 3 elbow: Δ axis / Δ coverage / mean F |
|---|---|---|
| axis gradient, translation only | +17.2 ± 0.3 mm / +0.004 / 37 N | +15.6 ± 0.2 mm / +0.022 / 43 N |
| rotation gradient only (2.5°/decision) | +4.8 mm / 0.000 / 36 N | +8.2 mm / +0.020 / 40 N |
| **5-D axis gradient** (translation + rotation) | +21.2 ± 0.2 mm / +0.015 ± 0.001 / 43 N | **+24.9 ± 0.0 mm / +0.077 ± 0.000 / 45 N** |
| axis gradient + the expert's own rotation | +17.3 mm / +0.004 / 39 N | +16.0 mm / +0.025 / 43 N |
| expert (96 / 88 mm executed, 0.04 / 0.19 rad) | +6.1 mm / **+0.027 ± 0.000** / 61 N | +3.9 ± 0.5 mm / +0.024 ± 0.004 / 69 N |
| minus force gradient | +10.8 mm / 0.000 / 40 N | +6.2 mm / 0.000 / 40 N |
| random (3, drawn per state) | ≤ +8.9 mm / 0.000 | ≤ +9.6 mm / ≤ +0.035 |

Reading. Rotation is not a detail: adding the rotation part of the axis gradient to its
translation part raises the coverage gain from +0.004 to +0.015 at cell 1 and from +0.022 to
**+0.077** at cell 3, where the 5-D proxy gradient walks the sleeve over the elbow with three
times the expert's coverage gain at half its travel and two thirds of its load. At cell 1 the
same walk stays below the expert (+0.015 against +0.027): the expert's gain there comes from a
translation direction, (0.25, 0.90, −0.36), that the coverage reward cannot rank (flat) and the
axis proxy points away from. So one of two quiet elbows is solved by a one-step 5-D gradient of
a proxy, the other is not; the elbow needs the critic for generality, and the 5-D proxy gradient
is the baseline Level 3 has to beat there.

## Decisions taken for Levels 2 and 3 (2026-09-14, delegated by the owner) [E]

- **The tether and the no-move collision rule stay as they are.** Both come from the reference
  environment; smoothing them into penalties would change the task against which every number in
  this directory was measured. Level 2 differentiates with respect to the *executed* command, and
  Level 3's actor is trained on the executed trajectory; where a command is refused the executed
  derivative is zero on that side, which is the truth of the task.
- **Gradient use is gated per state by a repeated decision, not by solver statistics.** Cell 2's
  elbow scattered 23 mm with one Newton iteration; cell 1's 0.8 mm with three. The gate is the
  spread of one repeated decision from the same restored state (one extra decision per state), and
  a state that fails it contributes no physics gradient, only samples.
- **Rotation is probed before the actor is designed.** The environment executes two rotation axes
  (about x is clipped, 5° per decision at most); the elbow's coverage gain in the expert's hands
  came with rotation, so the probe's rotation differences and rotation walks (below) decide whether
  the elbow needs the critic or a one-step 5-D direction.

## What Level 1 says about Levels 2 and 3

The quantity Level 2 would compute analytically — how the cloth state after a decision moves with
the gripper command, `∂x⁺/∂u` — is well defined and repeatable at every deterministic state,
including the two quiet elbows (the axis reading, a function of `x⁺`, has a repeatable gradient
there). What is flat at the elbow is the *reward* as a function of `x⁺`, not the state's
sensitivity to the command. So the split has to be SHAC's: the physics supplies `∂x_{t+h}/∂u_t`
through the solver, a learned value supplies `∂V/∂x_{t+h}` and bridges the plateau — a critic
can rank elbow states by how close they are to passing even where the reward cannot. A greedy
one-step gradient controller (DiffCloth's use) would be enough after the elbow, would need a
hand-written release phase at a lock, and is useless at the elbow; the elbow is where our policy
fails, so Level 3, not Level 2 alone, is the deliverable. The lock re-probe fixes the horizon's
lower bound: at least the four decisions of a release phase, i.e. h ≥ 4 decisions = 24 frames.

Level 2 in the backend, minimal form, to be validated against this probe before anything is
trained: after a frame's Newton loop converges, keep its assembled system (`GlobalLinearSystem`'s
`bcoo_A`, the one `CurrentFrameDiffDofReporter` already reads) and solve `H λ = g` for a caller's
`g` with the same PCG and preconditioner; the command enters through `SoftPositionConstraint`
whose cross term is the closed form `−s·m·ratio·I` on the held vertices' rows, so one solve per
frame gives `gᵀ ∂x⁺/∂u`. The six frames of a decision chain through the inertia term (each frame's
`x̃` depends on the previous `x` and `v`), which is DiffIPC's reverse pass: one extra solve per
frame, against a forward solve that after the elbow is a single Newton iteration. The chain must
be built along the *executed* command: the tether and the no-move collision rule are maps from
the commanded to the executed action that no adjoint sees, and Level 3 either differentiates with
respect to the executed command or replaces those rules by smooth penalties. Acceptance test:
the analytic `∂coverage/∂u` at cell 1's passed state and cells 2–3's stalls must match the 1–2 mm
central differences above to cosine ≥ 0.95, and the two noisy elbows must be rejected by the
same repeatability gate.

## Level 2a: the adjoint through the solver's own Hessians, against differences (2026-09-14) [MI]

No backend change. The engine's `extras/debug/dump_linear_system` switch (present in the 0.0.28
wheel) writes every Newton iteration's assembled system to Matrix Market files under
`<workspace>/debug/cuda/linear_system/global_linear_system.cu/`; the files hold the upper block
triangle under a `general` header. `python -m uipc_manip.physics_gradient_adjoint` drives the
expert to a state with the dump on (discarding files after every decision), restores it, runs one
hold decision, reads the six frames' last Hessians back (11,679 DOFs: a 12-DOF fixed arm body
and 3,889 cloth vertices; 452k nonzeros; 3.6 MB each; scipy LU 0.3 s each), and runs

- the reverse pass `H_f λ_f = ĝ_f`, `ĝ_6 = ∂L/∂x_6`, `ĝ_f = 2Mλ_{f+1} − Mλ_{f+2}` (BDF1 inertia,
  `x̃_f = 2x_{f−1} − x_{f−2} + g dt²`, lumped masses density × the backend's vertex volumes), with the gripper entering
  through the soft position constraint on the 48 held vertices, `∂L/∂Δ = Σ_f (f/6) Σ_I K_I λ_{f,I}`;
- the tangent pass, the same linear algebra forward, giving the full response `∂x_6/∂Δ` of every
  cloth vertex to each translation axis, compared with central differences of every vertex position.

Cell 3's stall (the expert's `align_yaw` stall, 26 N, two Newton iterations per frame), two draws
of the state (the expert's re-drive lands within 0.2 mm of itself), 1 mm and 2 mm differences:

| quantity | reverse pass vs differences (cosine / magnitude ratio) | note |
|---|---|---|
| held vertices' own response to their aim | **1.000 / 0.997–1.000** (they follow 0.999 of a commanded metre, predicted 0.995–0.998) | with the backend's own vertex volumes (the correction below) |
| free cloth response field (all other vertices) | 0.88–0.97 / 0.96–1.02 per axis in one draw, 0.71–0.97 / 0.75–1.02 in the other | the six-frame chain beats the last frame alone by 0.01–0.17 in cosine |
| upper-arm axis reading (linear objective) | **0.989–0.993 at 1 mm, 0.987–0.997 at 2 mm** / 0.83–0.85, 0.73–0.95 | passes the 0.95 gate on direction |
| contact energy | 0.65–0.76 at 1 mm, 0.56–0.75 at 2 mm | its own differences disagree between 1 and 2 mm by up to 6×: not a smooth function of the command at this scale |

Two things the check settled that the source reading had not, one of them wrongly at first.
(i) **`thickness` is a half-thickness in libuipc.** The first pass lumped the cloth's mass as
density × thickness × area / 3 and found the held vertices responding to exactly twice the
predicted amount (0.498 against 0.999 per commanded metre); it blamed a doubled constraint
stiffness. The backend's own vertex volume is twice that (`src/geometry/compute_vertex_volume.cpp`
takes `h = 2 r` for a shell), so the mass is twice that, and the constraint's assembled block is
exactly `s·m·I` with the backend's `m`: `uipc_test_diff_sim` differences the held vertex's exported
diagonal block between strengths 100 and 200 on a tetrahedron at rest and gets `100 · m · I` to
1e-6, no time-step factor, no double count. The script now reads the `volume` attribute the
constitution wrote instead of recomputing it (`--constraint-factor` is 1). The same convention makes
the environment's cloth (`cloth_thickness` 0.15 mm) twice as heavy as the parameter reads; nothing
measured in this directory depends on it, but it is the value to quote. (ii) **With the right masses
the free cloth's response is reproduced in magnitude as well** (0.96–1.02 per axis in one draw at
cell 3, 0.75–1.02 in the other): the 20–25 % shortfall the first pass attributed to the
positive-semidefinite projection of the assembled Hessian was the halved inertia coupling in the
chain. What remains short with the right masses is the *objective*: the axis reading's predicted
magnitude is 0.64–0.85 of the measured one at every state while the bulk field is within a few
percent of one, so the opening's vertices — the ones in contact with the arm — respond 15–35 %
less than the linear model says and the free bulk does not. That is where the projection
(28 `make_spd` sites) and friction's lagged tangent basis act, and it is the part of the cloth
Level 3's gradient acts on; direction passes, the scale is the learner's to set. Where the linear
model fails outright is the jam (below).

The field check is noise-limited. A 1 mm command moves a free vertex by 0.2–0.3 mm RMS, while two
identical decisions from the same restored state differ by 0.07–0.14 mm at cell 3 (measured in the
Level 2b check below), so a field cosine near 0.9 is close to the ceiling this simulator allows;
the objective-level readings sum the opening's vertices and are less exposed.

Cost on this mesh: six exports (1.5–2.2 s) and six host factorisations (1.7–3.7 s) per decision;
the backend's own solve reaches 1e-6 in 0.12–0.20 s (Level 2b), and a stored matrix is 3.6–7 MB,
so a device-side ring of the last 24–72 frames (a four-to-twelve-decision horizon) is 100–500 MB.

**Two more states.** The same check at cell 2's stall (the jam at 51 N, 20,523 DOFs, three
Newton iterations per frame) and cell 1's passed state (25 N), all on this tree's CUDA build
through the Level 2b export (the exported systems are the dumped ones, below; the wheel's dump
path had given the same objective cosines within 0.01 before the mass correction):

| state | axis reading: cosine at 1 / 2 mm (magnitude at 1 mm) | contact energy: cosine at 1 / 2 mm | free cloth field: cosine / magnitude per axis | held vertices |
|---|---|---|---|---|
| cell 3 stall, two draws (26 N) | **0.989–0.993 / 0.987–0.997** (0.83–0.85) | 0.65–0.76 / 0.56–0.75 | 0.71–0.97 / 0.75–1.02 | 1.000 / 0.997–1.000 |
| cell 2 stall, the jam (51 N) | **0.992** / 0.935 (0.72) | **0.97** / 0.72 | x 0.14, y 0.23, z 0.92 / 0.61, 0.40, 0.90 | 1.000 / 0.99–1.00 |
| cell 1 passed (25 N) | **0.995 / 0.997** (0.64) | 0.89 / **0.99** | 0.84–0.94 / 0.91–1.11 | 1.000 / 0.99–1.00 |

The smooth objective's direction passes the gate at every state at 1 mm (0.989–0.995) and in
three of four draws at 2 mm; its magnitude is 15–35 % short, most at cell 1's passed state. The
contact energy passes only where it is itself smooth. The free cloth's full response is
reproduced where contact is light (cells 3 and 1: cosines 0.71–0.97, magnitudes within 25 % of
one) and not at the jam, where two of three axes come back at 0.14 and 0.23: there the measured
1 mm response of the free cloth is contact rearrangement, partly noise (the state repeats within
0.2 mm) and partly what the projected Hessian does not carry, while the objective's direction
still is. The linear model's cost is state-dependent and largest exactly where the load is, which
is what the per-state repeatability gate is for.

## Level 2b: the export and the solve inside the solver (2026-09-14) [MI]

`LinearSystemAdjointFeature` (`diff_sim/linear_system_adjoint`; header
`include/uipc/diff_sim/linear_system_adjoint_feature.h`, CUDA side
`src/backends/cuda/linear_system/linear_system_adjoint.cu`, Python
`uipc.diff_sim.LinearSystemAdjointFeature`) exposes, after every `World.advance()`, the system the
frame's last Newton iteration assembled: `export_system()` copies `GlobalLinearSystem`'s
block-sparse `bcoo_A` (the upper block triangle after symmetric compression, 3×3 blocks) and its
gradient `b` to host arrays, and `solve(rhs, rel_tol, max_rounds)` runs the frame's own PCG with
its preconditioner on a caller's right-hand side by iterative refinement (`x += solve(rhs − H x)`
with the backend's own sparse product) until the relative residual reaches `rel_tol`, then puts
the frame's `b` and `x` back. Refinement is not optional: the environment runs the frame solver at
`linear_system/tol_rate` 1e-2 (an inexact Newton step), and a single solve on a random right-hand
side of the dressing system came back with relative residual 3.6; refined, it reaches 2–3e-7 in
0.12–0.20 s on the 11.7k- and 20.5k-DOF systems and agrees with the host factorisation to 2–3e-8.
`apps/tests/diff_sim/linear_system_adjoint.cpp` (`uipc_test_diff_sim`, 174 assertions) checks the
export's symmetry and finiteness, the refined solve against the exported matrix, that the frame's
gradient is restored and a second solve repeats the first, the size checks, that the world still
advances, and the constraint-stiffness differential above. It is a module-loading test target
(like `sim_case`) because `backend_cuda`, which links the backend's objects into the binary, cannot
host a `World`: every kernel exists twice and launches fail with "invalid resource handle".

Checked on the dressing scene in one process (`--source feature --compare-dump`, twice at cell 3's
stall): the six exported systems are the six dumped ones to the bit (same block pattern, 452–453k
nonzeros, zero difference in every value and in the gradient), and a solve between two decisions
leaves the second within the run-to-run scatter (0.14 mm against 0.07–0.09 mm between two plain
repeats; one sample each). Export costs 0.25–0.37 s per frame (1.5–2.2 s per decision: host copy
and block expansion), the host factorisation 0.3–0.6 s per frame, so a decision's six systems
cost 3–6 s to export and factorise. That sets the size of the first Level 3 experiment; a
device-side ring of past frames' matrices would remove the export. ~33 shared-GPU minutes.

## Level 3, first experiment: open-loop trajectory optimisation at the two quiet elbows (2026-09-14) [MI]

Before an actor is trained, the question the advisor set: the one-step coverage gradient is zero
at the elbow; is the multi-step gradient of the objective after `h` decisions not, and does
walking it beat the 5-D proxy walk and the expert? `python -m uipc_manip.physics_gradient_trajopt`
restores an elbow state, takes `h = 12` decisions (72 frames) exporting every frame's system
through the Level 2b feature, and improves the 12 actions by projected gradient ascent (largest
component moved by 0.15 action units per accepted step, halved on rejection, stop below 0.02) on a
terminal objective in metres: the axis reading plus the reference's coverage distance (the
shoulder→elbow ray's first sleeve triangle, with its exact hit-triangle gradient), so the objective
is smooth before the sleeve reaches the upper arm and is the reward's own distance after. The
gradient is the reverse pass of all 72 systems (host factorisation as each is reached, 21–42 s)
with the commands entering through the held vertices' aims — translation through the anchor,
rotation through the offsets (`δaim = δθ × offset`, carried to later frames by the rotations
executed in between) — with respect to the executed command. Initial trajectory: the 5-D proxy
walk of the rotation probe (4 mm and 2.5° per decision along its directions). Baselines from the
same restored state: the closed-loop expert and hold. `tests/test_physics_gradient_trajopt.py`
(4) checks the chain's last decision against the one-decision adjoint, its command bookkeeping
against differences of the linearised aim model, that refused substeps contribute nothing, and
the coverage gradient. One rollout with export takes 12–20 s, one reverse pass 21–42 s.

| elbow | 5-D walk (it 0) | optimised, mean ± std of 3 repeats | expert (12 decisions) | iterations |
|---|---|---|---|---|
| cell 3 (`tshirt_26`/14049 @93) | L 0.046, coverage 0.073, 50 N, 48 mm | **L 0.100 ± 0.001, coverage 0.167 ± 0.001**, 74 N, 104 mm, 56° | L 0.014, coverage 0.028, 68 N, 88 mm | 9 accepted, 1 trial each, then no step |
| cell 1 (`tshirt_392`/14046 @86) | L 0.027, coverage 0.018, 50 N, 48 mm | **L 0.077 ± 0.002, coverage 0.106 ± 0.002**, 48 N, 105 mm, 50° | L 0.014, coverage 0.028, 63 N, 88 mm | 8 accepted, 1 trial each, then no step |

Both elbows are walked over: 6× and 3.8× the expert's coverage after twelve decisions, at cell 1
with less force than the expert (48 against 63 N) and at cell 3 with a little more (74 against
68 N); every accepted step was the first trial, and the three final repeats agree to 0.002 of
coverage, so the gains over the expert are established, not one draw. What part of them is the
gradient's is the control below.

**What the gradient actually is over twelve decisions.** The twelve rows of `∂L/∂a_t` came out
identical to three digits at both elbows (norm 0.0048 at every decision at cell 3, 0.0025 at cell
1). That is the structure of the chain, not a bug (the CPU tests cover the bookkeeping): the BDF1
inertia coupling `2Mλ_{f+1} − Mλ_{f+2}` carries a vertex mass of 4·10⁻⁴ kg against constraint
and elastic stiffnesses of order one, so `λ_f` decays within one or two frames of the end (the
frames before the last add ~13 % at the objective level in Level 2a) and the derivative with
respect to an early decision reduces to the *static* sensitivity of the final state to a rigid
shift of the whole later gripper path. The cloth is quasi-static at this time step; what the
linear chain cannot see is path dependence through contact and friction states. Central
differences of `L` after twelve decisions with respect to single decisions (2 mm, 1°, one draw
each, at the 5-D walk trajectory) show exactly that: at cell 1 the translation row agrees in
direction at lags 11, 5 and 0 (cosine 0.96, 0.99, 0.70) at 0.44–0.67 of the measured magnitude,
the rotation row at lag 11 only (0.89; 0.41 and 0.36 at lags 5 and 0); at cell 3 the first
decision's difference points the other way (translation cosine −0.37, rotation 0.83 at 0.08 of
the magnitude): moving the first decision alone changes what the sleeve catches on, which no
fixed-active-set linearisation carries. So the optimiser, fed twelve equal rows, moved all twelve
actions together: at both elbows it took the translation to the action box (8.7 mm per decision,
from 4 mm) along nearly the proxy's direction and turned the rotation axis from (−0.3, 0.95) to
(−0.47, 0.88) at 4.2–4.6° per decision (from 2.5°). Whether that gain is the scale or the direction
is the control below.

**Lag check at cell 3, two draws of every difference** (2 mm, 1°; the 5-D walk trajectory; "draws"
is the cosine between the two difference draws, the noise floor of the reading itself):

| decision (lag) | translation: cosine / magnitude ratio / draws | rotation: cosine / magnitude ratio / draws |
|---|---|---|
| 1 (11) | **−0.30** / 0.20 / 0.86 | 0.98 / 0.34 / 0.49 |
| 4 (8) | 0.99 / 0.61 / 0.98 | 0.94 / 0.38 / 1.00 |
| 8 (4) | 0.99 / 0.82 / 0.89 | 0.98 / 0.33 / 0.99 |
| 10 (2) | 0.91 / 0.70 / 0.94 | 0.96 / 0.59 / 0.46 |
| 12 (0) | 0.97 / 0.68 / 1.00 | 0.63 / 0.86 / 0.10 |

The static row is the right direction for the translation of every decision but the first (0.91–0.99
where the differences repeat, at 0.6–0.8 of the magnitude), and for the rotation wherever the
difference itself repeats (lags 8 and 4: 0.94–0.98; at lags 11, 2 and 0 the two draws of the
rotation difference disagree with each other, so nothing is read there). The first decision is the
exception at cell 3 and not at cell 1: its difference is repeatable (0.86) and points against the
chain — moving only the first 8 mm of the approach changes what the sleeve catches on, which no
fixed-active-set linearisation carries.

**Control: the proxy direction at the action box, no optimisation** (the 5-D walk's directions at
8.66 mm and 5° per decision, three repeats; a fresh re-drive of each elbow, @94 and @84):

| elbow | scaled proxy, no gradient | optimised (above) | expert |
|---|---|---|---|
| cell 3 | coverage 0.162 ± 0.001, 83 N | 0.167 ± 0.001, 74 N | 0.028, 68 N |
| cell 1 | coverage 0.094 ± 0.000, 47 N | 0.106 ± 0.002, 48 N | 0.028, 63 N |

Most of the gain over the expert is the scale: the one-step 5-D proxy direction of the rotation
probe, driven at the action box instead of 4 mm and 2.5°, already beats the expert 3.4–5.8× in
coverage at both elbows. The optimiser's own contribution had to be measured from the *same*
restored state, because two re-drives of cell 3's elbow gave the identical 4 mm walk 0.073 and
0.086 of coverage — a re-drive spread of 0.013 that the within-state repeat std of 0.002 does
not see. Same state, three rollouts each (`--init actions-json --baselines expert scaled_proxy`):

| elbow (same restored state) | scaled proxy, no gradient | optimised trajectory | expert |
|---|---|---|---|
| cell 3 @93 | 0.163 ± 0.001, 73 ± 4 N | **0.170 ± 0.001, 61 ± 6 N** | 0.031, 76 N |
| cell 1 @86 | **0.107 ± 0.001, 42 N** | 0.105 ± 0.001, 52 ± 3 N | 0.028, 63 N |

So the twelve-decision gradient adds +0.007 of coverage and takes 12 N off the force at cell 3,
and adds nothing at cell 1 (−0.002, inside the spread) while costing 10 N; the +0.012 the first
control suggested there was re-drive spread. It certainly does not find a *sequence*: with twelve
equal rows it cannot, and the states where a sequence is needed (the lock) were kept out on
purpose.

**What this says about the actor.** (i) The physics gradient's useful horizon at this time step is
about one decision: beyond it the chain is the static sensitivity, which is right in direction
at most lags but carries no path dependence. SHAC's premise — a long differentiable horizon —
does not hold for quasi-static cloth in frictional contact; the split that fits the measurements
is SVG(1)-like: the solver's one-decision Jacobian `∂x'/∂u` (six exported systems, or six device
solves against them) times a learned `∂V/∂x'` from a TD critic, with the critic carrying
everything beyond one decision. (ii) The state sensitivity is there at the elbow with the right
sign at every decision but the first of a fresh approach, so an actor gradient through it is not
signal-less where the reward is flat, which was the point of the line. (iii) Cost: a one-decision
Jacobian is 1.5–2.2 s of export plus 1.7–3.7 s of host factorisation, or six refined device
solves at 0.12–0.20 s (0.7–1.2 s), against the training loop's 0.220 s of simulation per
transition (13,881 transitions per hour): device solves on every transition are a 3–5×
slowdown, the host path 14–27×. So the actor experiment uses the device solves, applies them to
a subset of transitions (the ones that pass the repeatability gate, or one in four), and is
sized as minutes on the two elbows first, not as a training run. ~85 shared-GPU minutes for
this section.

## Level 3, second experiment: the critic says where, the adjoint says how (2026-09-14) [MI]

`python -m uipc_manip.physics_gradient_actor` takes a trained checkpoint's critic as the value
of the next state, `V(x') = min(Q1, Q2)(s', μ(s'))`, differentiates it exactly through the
environment's observation function (camera visibility, 6.25 cm voxel centroids, tool-relative
packing — rebuilt in torch from the discrete choices the environment made at `x'`: 1,300–2,300
visible vertices in 130–180 voxels), and multiplies by the one-decision adjoint:
`∂V/∂u = (∂V/∂x')ᵀ ∂x'/∂u`, from the six exported systems on the host (1.8–3.9 s) or the last
frame's device solve (0.10–0.21 s). Checkpoint: the dense-critic ablation `abl_dense_s1` at
125k updates (Wang flow actor, trained on the region's 45 bodies × 5 garments; the probed bodies
14046 and 14049 are its evaluation cells). Four states: cell 3's elbow and stall, cell 1's elbow
and passed state. `tests/test_physics_gradient_actor.py` (3) covers the torch observation (it
reproduces the centroids and spreads a centroid's gradient over its visible members), the
last-frame command mapping and the action scaling.

Two readings. (i) Against 2 mm / 1° central differences of `V` after one decision, the analytic
`∂V/∂u` agrees only loosely: cosine 0.60–0.71 at a hold decision where the differences repeat
(draw-to-draw cosine 0.86–1.00), negative where they do not (at the policy's own action the two
draws of the difference agree with each other at −0.14 to 0.61), and its magnitude is 0.05–0.19
of the measured change. The critic is not a smooth function of the cloth at the millimetre
scale: a 2 mm command moves vertices across voxel boundaries and in and out of visibility, and
the point network is rough besides, so most of the measured change of `V` is not its derivative.
(ii) What an actor needs is a direction that improves the state when followed. Twelve greedy
decisions at the action box (8.66 mm, 5°), the direction recomputed at every step, five signals
from the same restored state — a walk at the box tests a direction's sign pattern, not the
gradient's magnitude, and the proxy walk is the reference for what any up-arm direction achieves:

| state | physics: `∂V/∂x'` through the adjoint | SAC's own `∂Q/∂a` | the checkpoint's policy `μ` | hand-written 5-D proxy | expert |
|---|---|---|---|---|---|
| cell 3 elbow @94 | **0.168**, V 75→88, 47 N | 0.114, V→86, 46 N | 0.076, 41 N | 0.163, V→79, 90 N | 0.046, 66 N |
| cell 3 stall @116 | 0.172, V 80→87, 33 N | 0.047, 50 N | 0.040, 22 N | **0.214**, 36 N | 0.051, 33 N |
| cell 1 elbow @84 | **0.135**, V 73→80, 34 N | 0.000 (−11 mm), 34 N | 0.016, 44 N | 0.076, 46 N | 0.028, 57 N |
| cell 1 passed @125 | 0.151, V 73→79, 59 N | 0.147, 29 N | 0.054 (−21 mm), 28 N | **0.220**, 41 N | 0.146, 257 N |

(coverage after twelve decisions; `V` the critic's value at the end; force the mean net normal
force over the walk; one draw per walk, and a greedy walk recomputes a rough direction at every
step, so the fixed-trajectory repeat spread of 0.001 does not apply to it.) The physics signal
beats the critic's action derivative at the policy's action at three of four states — by 0.05 at
cell 3's elbow and by 0.135 at cell 1's, where SAC's direction retreats — and by 0.004 in one
draw at the passed state; it beats the checkpoint's policy and the expert everywhere, beats the
hand-written proxy at cell 1's elbow, the state the proxy could not solve (0.135 against 0.076),
and matches it at cell 3's elbow at half the force. After the elbow the proxy, which is the axis
reading's own gradient, is the better greedy signal (0.214 and 0.220 against 0.172 and 0.151).
What is measured is that `V`'s ascent direction, taken through the physics, also raises the
coverage at the elbow where the reward is flat; the proxy walk reached the same coverage at cell 3
with `V` at 79 against 88, so `V` is not a ranking of elbow states, only a usable direction. That
`∂Q/∂a` was read at the policy's action, where a converged actor's action derivative is small and
its direction noisy; the like-for-like comparison is `∂Q/∂a` at the hold action, the learned
estimate of the same quantity the adjoint computes at the hold decision. Followed greedily the
same way (`--walks sac_hold`, a fresh re-drive of each state):

| state | adjoint at hold (above) | `∂Q/∂a` at the hold action |
|---|---|---|
| cell 3 elbow | **0.168** | 0.000 (retreats 15 mm) |
| cell 3 stall | **0.172** | 0.018 → 0.000 (retreats 13 mm) |
| cell 1 elbow | **0.135** | 0.000 (retreats 3 mm) |
| cell 1 passed | 0.151 | **0.184** |

At the three states at or before the elbow the critic's learned action dependence, read at the
same action, walks backwards, and the solver's Jacobian applied to the same critic walks over;
after the elbow the learned one is the better of the two. That is the decisive reading for the
split: the critic knows the direction in state space at the elbow and not, at that point, in
action space, and the adjoint converts the one into the other. One caveat: the hold action is
one the critic was rarely trained on, so its `∂Q/∂a` there is an extrapolation; at cell 1's
elbow the same conclusion holds at the policy's own action (0.000 against 0.135), so the reading
does not rest on it.

**Where the gap between `V`'s differences and its derivative comes from, measured.** The same
differences taken through the observation with its discrete choices frozen at the unperturbed
decision (the visibility and voxel membership of that state, so only the differentiable part of
the observation moves) raise the magnitude ratio of chain to difference from 0.03–0.05 to
0.09–0.34 at cell 1 (draw-to-draw cosine 0.76–1.00) and agree with the chain in direction at
0.32–0.83 there; at cell 3 the plain and the frozen differences disagree with themselves between
draws at the elbow (−0.74 to 0.65) and nothing is read. So the observation's voxel and visibility
switches are a factor 2–7 of the 20–30× gap, and the rest is the critic network's own roughness
over 2 mm (the chain reproduces a smooth objective at 0.64–0.85 of its magnitude, Level 2a). The exception that stretches "poor precision, usable
direction" furthest is cell 3's stall: there the 2 mm differences repeat (plain 0.99, frozen
1.00) and the chain's direction is neither of them (cosine 0.01 and −0.04, magnitude ratio 0.015
and 0.04, the smallest of all states), yet the greedy walk along the chain from that state gained
coverage 0.007 → 0.172. At the action box a walk tests a direction's sign pattern, not the
gradient's magnitude; the proxy walk is the reference for what any up-arm direction achieves.

**The same experiment with a weaker critic.** The region's earlier teacher checkpoint (relaunch
2 at 14k updates, latent critic — the reference's rejected architecture, which this loader
accepts) inverts the picture: the physics walk retreats at both elbows (0.000 and 0.000) and
reaches 0.089 and 0.115 at the two post-elbow states, while its `∂Q/∂a` walks 0.169 / 0.207 /
0.118 / 0.164 (at μ) and 0.141 / 0.181 (at hold, cell 1). Along the physics walk this critic's
own value *fell* (21.3 → 20.5 at cell 3's elbow) where the dense critic's rose (75 → 88): its
`∂V/∂x'` is not a usable local ascent direction of its own value, so there is nothing for the
adjoint to convert. One checkpoint at 14k updates against one at 125k is a critic-quality
contrast, not a trend. ~1.3 shared-GPU hours for this section (same-state control, three actor
runs, two failed attempts, the reproduction).

**What this says about the actor.** The pieces of an SVG(1)-style update exist and work in the
direction that counts: a TD-trained point-cloud critic supplies a `∂V/∂x'` that, pushed through
the solver's one-decision Jacobian, walks both elbows over where the same critic's `∂Q/∂a` at
the same action, the policy it trained and the scripted expert do not — with the dense critic at
125k updates, and not with the 14k latent one. What is missing is the smoothness the update would
assume: `V`'s differences over 2 mm are mostly not its derivative, a factor 2–7 of that being the
observation's voxel and visibility switches and the rest the network, so a learner should treat
`∂V/∂u` as a stochastic direction (small steps, averaged over decisions, behind the repeatability
gate), not as a Newton step. The actor update proper — `μ_θ` moved along `∂V/∂u · ∂μ/∂θ` on gated
transitions inside the SAC loop — is a training run, not a probe, and is not started.

## What this does not claim

No policy has been trained with a physics gradient. Level 1 asks only whether the quantity is
well defined and locally informative at the states that matter; Level 2 (the adjoint through the
solver's systems, and the export and solve inside the solver) is built and checked against
differences at four states; Level 3's first experiment is open-loop trajectory optimisation at
two elbow states with the chain as its gradient, not a learner, and most of its gain over the
expert is the proxy direction driven at the action box; its second experiment follows a trained
critic's gradient through the one-decision adjoint greedily for twelve decisions, which is a
controller, not a learner, and reads one draw per walk. The field-level agreement is bounded by
the simulator's own run-to-run scatter, the linear model misses the free cloth's response at a
jam, and over many decisions it reduces to a static sensitivity that carries no path dependence.
Nothing here is a claim about a closed-loop policy, about states other than the seven probed, or
about the real robot.
