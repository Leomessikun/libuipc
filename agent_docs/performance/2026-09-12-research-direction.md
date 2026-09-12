# The research direction

Date: 2026-09-12. Derived from eight research tracks and our own measurements.

## In one sentence

**Contact force inside the online simulation training loop for robot-assisted dressing — the one
arrangement every published system in this field names as impossible — together with the first
measurement of whether the accuracy of that force matters to policy learning.**

## Why it is open, in the authors' own words

FMVP (CoRL 2025), explaining why its simulation stage carries no force: "this simulation training
does not use any force information, **as current simulators lack realistic force modeling for
deformable garments**" [RA]. FCVP (RA-L 2024) states its simulated threshold in "units" with the note
that they "do not correspond to Newtons in the real world" [RA].

| Work | Where force enters | Where the force comes from |
|---|---|---|
| Wang, RSS 2023 | a penalty term in the reward | a FleX proxy, arbitrary units |
| FCVP, RA-L 2024 | an execution-time action filter | 264 real trajectories |
| FMVP, CoRL 2025 | conditioning the policy, in real-world fine-tuning | real sensors, 192 trials |
| GentleHumanoid, 2025 | teacher observation, reward, impedance reference | a hand-built spring model |
| **online simulation training** | **nobody** | **no simulator reports it** |

Our simulator reports it: exact per-vertex normal force in newtons, verified against analytic answers
to six digits in three independent scenes [MI], separable per body and per contact type, on every step
of every environment, on failing episodes as well as succeeding ones.

## Three claims, ranked by how defensible they are

### 1. The measurement claim — safest, and already half-built

**Peak local contact pressure over roughly one square centimetre, per body segment.** The dressing
field has no shared definition of the quantity it bounds: two groups quote 18 N and 120 N for
different things, published real dressing uses 10 N, 1.02 N and 5.4 N with no standard cited, and
ISO/TS 15066's own table is force *and* pressure per body region measured on 100 healthy 18-to-66
year olds, which is an extrapolation to a frail arm. Force over a nominal area under-reports badly:
in the study behind that table an index fingertip carried 58 N median force but 185 N/cm^2 median peak
pressure [RA].

A wrist force reading cannot produce this. Per-vertex contact force can. This is a contribution
whether or not any policy improves.

### 2. The training claim — the actual research

**Force as a conditioning input to the policy and a signal the critic reads, inside online
reinforcement learning in simulation.** Two independent controlled comparisons fix the mechanism:

- FMVP re-tested FCVP's predict-and-filter on the same task and it lost under arm motion, concluding
  that conditioning on force and optimising the task directly is more robust [RA].
- RoboPack found the same shape on tactile: as an input 16 of 20, without it 8 of 20, and **predicting
  it 6 of 20, below the no-tactile ablation** [RA].

So the design is conditioning, not prediction. Nobody has done it in simulation because nobody's
simulator supplies the signal.

### 3. The fidelity claim — the sharpest, and the cheapest to test

**Does the accuracy of the training-time force matter?** GentleHumanoid dismissed physics-engine
contact forces as "often noisy, local, and uncoordinated", used a hand-built spring model, and
reported results anyway [RA]. Nobody has tested the assumption either way.

We can, and nearly for free, because we hold both ends:
1. the exact per-vertex IPC force, which only this class of simulator gives;
2. a crude spring proxy already in the loop — the cuff's hold is a spring of known stiffness and its
   displacement is in every info and in the privileged state [MI]; this is GentleHumanoid's choice
   reproduced exactly;
3. no force, which is Wang's setting.

Same architecture, same budget, three signals. **If the exact force ties with the proxy, the field's
implicit assumption is wrong and that is worth publishing. If it wins, it is the first measurement of
how much contact-force accuracy buys a policy.** The experiment is pure simulation, so none of the
sim-to-real doubts below touch it.

## What is explicitly not novel

- A force penalty in the reward: Wang has it, and Clegg measured 0 per cent task success training
  from scratch with one [RA].
- Filtering actions by predicted force at execution: FCVP, and its own successor re-tested it and
  found it weaker.
- Conditioning a policy on force during real-world fine-tuning: FMVP.
- A privileged-force teacher with a force-blind student: GentleHumanoid, on a real humanoid.
- Reading forces out of an IPC solver at all: TaCauchy does it on this very codebase, for tactile
  sensors, validated against real responses at SSIM above 0.93 over 1.3 to 4.7 N [RA].

## The risks, stated plainly

- **Simulation is wrong where it is unique.** The garment catches that dominate real force are the
  ones Yu et al. had to exclude from a parameter fit because they drove friction to unrealistic
  values, and FCVP states they "usually do not occur in simulation" [RA]. Our force is exact for our
  contact model; that is not the same as being right about a real forearm.
- **Shear is unbounded.** The friction channel is unusable as exported [MI], and shear is the named
  mechanism of skin injury. Everything here bounds normal pressure only. Fixing the friction readout
  is separate work and a possible second contribution.
- **The deployment channel is coarse.** A Stretch 3 has no force/torque sensor. The deployable
  quantity is the gripper-side wrench, which we already have free from the hold's stiffness times its
  tracking error [MI], and which on the real robot means joint effort — ForceSight's grip-force model
  from motor current reports 1.524 N RMSE [RA].
- **Transients.** Opening iterates reach three orders of magnitude above the physical value [MI], and
  settled readings reproduce to 3 to 5 per cent where transients differ ninefold [MI]. Every use of
  the signal has to be gated first.
- **No world model has ever controlled cloth beyond a horizon of one to five actions** [RA], so the
  sample-efficiency half of any plan is unproven on this material.

## The first three things to do

1. **Gate the transients on the engine's Newton telemetry, and ship both force channels into `infos`.**
   Half a day. Nothing below means anything without it.
2. **The pressure aggregator**: per-segment resultants and peak pressure per square centimetre, on the
   arm's contact force and on the gripper wrench. One day. This is claim 1, delivered.
3. **The three-way fidelity ablation.** About three days of compute. This is claim 3, and it decides
   whether claim 2 is worth a month.
