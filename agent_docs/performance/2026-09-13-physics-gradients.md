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

## What this does not claim

No policy has been trained with a physics gradient. Level 1 asks only whether the quantity is
well defined and locally informative at the states that matter; Level 2 (the adjoint in the
backend) and Level 3 (the short-horizon actor with a terminal critic) are not built.
