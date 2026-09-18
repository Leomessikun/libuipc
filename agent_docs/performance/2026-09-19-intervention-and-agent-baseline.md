# Full-episode recovery, what the failures actually are, and a first agent baseline — 2026-09-19

Status: three measurements complete, nothing trained, no solver, reward or controller
setting changed. They answer the question the
[recovery-decision study](2026-09-18-recovery-decisions.md) left open — whether a
recovery that helps locally helps a whole episode — and they replace the assumption
that the dressing policy fails by stalling.

Code: `uipc_manip.recovery_intervention`, `uipc_manip.agent_harness`,
`scripts/capture_policy_geometry.py`, `scripts/audit_progress_metric.py`,
`scripts/agent_programs/`. Artifacts under
`output/uipc_manip/recovery_intervention_20260919/`, `.../metric_audit_20260919/`,
`.../agent_harness_20260919/`.

## 1. The recovery never gets a chance: the policy does not stall

Three arms from the same eight seeds, 300 decisions each: the policy alone, the policy
with a stall detector that hands eight decisions to the garment's best macro, and the
same detector handing them to a macro drawn at random. The detector reads only what
the robot can see: metres of garment-centroid motion per metre of commanded tool
translation over eight decisions.

| Cell | Arm | Success | Sustained coverage | Peak coverage | Whole-episode grasp valid | Detector firings per episode |
|---|---|---:|---:|---:|---:|---:|
| tshirt_26/14046 | control | 0/8 | 0.538 | 0.598 | 0/8 | 0 |
| tshirt_26/14046 | best macro (`lift`) | 0/8 | 0.550 | 0.609 | 0/8 | 0.25 |
| tshirt_26/14046 | random macro | 0/8 | 0.583 | 0.613 | 0/8 | 0 |
| tshirt_392/14046 | control | 0/8 | 0.000 | 0.623 | 8/8 | 0 |
| tshirt_392/14046 | best macro (`forward`) | 0/8 | 0.000 | 0.613 | 8/8 | 0 |
| tshirt_392/14046 | random macro | 0/8 | 0.000 | 0.615 | 8/8 | 0 |

The detector fired twice in 24 episodes on one cell and never on the other, so two of
the six arms are the control run repeated. The spread between arms — 0.538, 0.550,
0.583 sustained coverage — is the run-to-run spread of the simulator over eight
episodes, the same order as the 0.006–0.013 measured for single continuations. No arm
differs from the control by more than that.

The reason the detector is silent is that the garment keeps following the commands.
The policy commands a median 2.72 mm per decision, 57.8 % of decisions above 2 mm, and
the accepted translation is the commanded one. Nothing stalls.

## 2. What the failures are instead

**tshirt_26/14046 loses the grasp.** Every one of the 24 episodes exceeds the 2 cm
tracking limit, reaching 30–31 mm. Coverage plateaus at 0.54–0.58 and stays there; the
episode is lost to the grip, not to the sleeve's progress.

**tshirt_392/14046 slides off the arm.** Coverage climbs to 0.59–0.62 by decision 120
and reads zero from decision 227–250 onward in every episode, with the grasp valid
throughout. Recomputed from a fresh 300-decision geometry capture of the same policy
and seeds, at the moment it happens the opening ring's centre is 8.6–9.1 cm from the
arm's centreline while the ring's own radius is 9.0–9.1 cm. The ring is balanced on
the edge of the arm and then leaves it. This is not a stall and not a jam: the sleeve
is not caught, it is not contained.

## 3. The metric's one-decision jumps are a marginal state, not a bug

Coverage jumps between 0.59 and 0.000 in a single decision, sometimes back and forth.
The progress metric casts a ray along each arm segment and asks whether it hits the
opening, which is stored as a six-vertex polygon triangulated into four triangles, so
a ray passing near the rim is a coin toss. Recomputing the metric from saved geometry
and comparing it with the winding test and with the ring-centre distance:

| Capture | Decisions | Rays read zero | Of those, the ring still encircles the arm | Isolated one-decision zeros |
|---|---:|---:|---:|---:|
| audit capture 0 | 301 | 47 | 12 | 0 |
| audit capture 1 | 301 | 181 | 14 | 0 |
| audit capture 2 | 301 | 18 | 11 | 0 |
| audit capture 3 | 301 | 178 | 27 | 0 |
| tshirt_392 capture 0 | 301 | 63 | 6 | 0 |
| tshirt_392 capture 1 | 301 | 79 | 22 | 0 |

So 4–9 % of a typical episode's decisions report no progress while the opening still
encircles the arm: the metric is noisy at the margin and should not be read at a
single decision. But the large zero stretches are genuine — in the two captures that
end at zero, the ring's centre finishes 20.6 and 22.7 cm from the centreline against
ring radii of 14.1 and 14.3 cm. The sleeve really is off the arm. An earlier reading
of these jumps as a measurement artifact was too strong.

## 4. A first agent baseline: a program written against a robot's own tools

`uipc_manip.agent_harness` gives a program the tool surface the published
agentic-robotics demonstrations use: the segmented point cloud in the tool's frame,
the tool pose, the direction to the shoulder, the garment's motion per commanded
metre, a rendered view of the state, commands in metres clipped to the controller's
own per-decision limits, and the option to hand decisions back to the trained policy.
Task metrics stay behind `report()` for the evaluator. Every call is logged.

Two programs, both on tshirt_26/14046 from the same seed as the policy arms above:

| Program | Decisions | Peak coverage | Max tracking error | Grasp valid |
|---|---:|---:|---:|---:|
| v1: travel toward the shoulder, lift when the garment stops responding | 300 | 0.000 | 32.5 mm | no |
| v2: take the opening past the fingertip first, then travel along the arm's axis | 300 | 0.000 | 5.5 mm | yes |
| the trained policy, same cell and seed | 300 | 0.598 | 30.1 mm | no |

The rendered frames show what each program did. The first dragged the cuff to the
shoulder along the outside of the arm and never threaded it: by decision 150 the
opening sits at the shoulder with the arm outside the ring. The second, written after
looking at those frames, kept the grasp and stayed at the hand but never enveloped the
fingertip either. Neither reached any upper-arm coverage while the trained policy on
the same cell and seed reaches 0.598 at its peak.

This is two programs and one seed, written by this session in two revisions, so it
bounds nothing about what a patient agent could do; the harness exists for more. What
it does establish is that a first-principles program over deployable quantities does
not thread a sleeve, while a policy trained in the simulator partly does.

## Reading

The recovery-decision line is closed by measurement, not by argument: the policy does
not stall, so a stall-triggered recovery has nothing to fire on, and the macros that
win locally at decisions 100–180 leave the episode where it was. The two failures the
episodes actually contain are grasp retention and lateral containment of the opening,
and neither is addressed by any macro in the library.

Both are fast, contact-level quantities rather than decisions a slow reasoning layer
would make, which is consistent with what the public agent demonstrations show: no
catalogued case runs a loop above about 1 Hz, none involves cloth on a body, and on a
precision insertion task two different models stalled at the same contact step. The
next measurements follow from the two failures directly: what the grip does in the
decisions before the tracking error exceeds 2 cm, and what the opening's lateral
offset does in the decisions before it leaves the arm, both computed from quantities a
robot can see.

## Limits

Two garment/body cells, eight seeds, one policy, one detector threshold. The stall
detector was set from a threshold that never fired, so nothing here tests a
well-tuned stall detector on a policy that does stall. The metric recomputation uses
the default reward geometry, without the opt-in five-centimetre shoulder extension.
The agent programs are this session's own two attempts.
