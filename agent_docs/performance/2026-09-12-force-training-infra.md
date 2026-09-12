# The infrastructure to build: force inside the learning loop

Date: 2026-09-12. Status: design, grounded in what the published work structurally could not do.

Tags: [RA] reported by the authors, [MI] measured here, [E] our estimate.

## The gap, stated precisely

FCVP (RA-L 2024) is the state of the art for force in dressing, and **force is not in its learning
loop**. Read from the paper:
- Its base policy is SAC on segmented point clouds, trained in simulation, **with no force at all**.
- Its force dynamics model — PointNet++ over the point cloud, five steps of force history and the
  candidate action, predicting the next step's scalar force under an MSE loss — is trained
  **separately and supervised, on 264 real-robot trajectories from 11 participants** [RA].
- Force is used **only at execution**: sample candidate actions, discard those predicted above a
  threshold, execute the highest-probability survivor.

It is arranged that way because NVIDIA FleX reports no usable contact force, so the signal had to be
bought with real robot time, and real robot time is too scarce to put inside a training loop.

**We have that signal for free, in simulation, per vertex, at every step of every environment** [MI].
That is the whole opportunity, and it is narrow and specific: not "we have better physics", but "the
quantity their method had to purchase is an output of ours".

## What the precedents say

- **A privileged force signal training a force-blind student works on real human contact.**
  GentleHumanoid (Stanford, 2025) trains a humanoid this way: contact force enters the teacher's
  observation, three reward terms and an impedance reference; the deployed student on a Unitree G1
  observes only a scalar safety threshold and has no force sensor [RA].
- **A force model trained in simulation transfers to real people.** Deep Haptic MPC (2018) trained a
  force predictor on 10,800 simulated dressing trials and ran it on a PR2 with 10 participants over
  480 trials, reporting 98.75 per cent full-arm dressing at a 0.2 s horizon against 1.25 per cent at
  0.01 s [RA]. It sensed force at deployment through a wrist sensor.
- **Nobody shows that the accuracy of the training-time force matters.** GentleHumanoid states that
  physics-engine contact forces "are often noisy, local, and uncoordinated", uses a spring model as
  the reference force and the simulator's force as the actual, and reports results anyway [RA]. Its
  stated limitation is that "spring-based modeling lacks real contact complexity (friction,
  viscoelasticity)".

**So the claim our simulator makes — that an exact contact force is worth having — is untested.** That
is simultaneously the contribution and the risk, and it is cheap to test because we can produce both
an exact force and a crude one.

## The infrastructure

Five components. Two exist, three are new. None of it reproduces regional teachers or clones
demonstrations into a diffusion policy.

### 1. Force readout — built

`uipc_manip.contact_force`. In: a running scene. Out: per-vertex normal and friction force in newtons
for one body's index block. Verified against analytic answers to six digits in three scenes [MI].

### 2. Force aggregator — new, half a day [E]

In: per-vertex forces plus a map from arm vertex to body segment. Out, per decision:
- the resultant force per segment (hand, wrist, forearm, elbow, upper arm);
- **the peak local pressure over roughly one square centimetre**, which is the quantity the dressing
  field has no shared definition for and a wrist force reading cannot produce.

Gated on the engine's per-frame Newton telemetry, because transients reach three orders of magnitude
above the physical value [MI] and settled readings reproduce to 3 to 5 per cent where transients
differ ninefold [MI].

### 3. Force dynamics model — new, two days [E]

The same shape as FCVP's, trained on simulation instead of real trials. In: point-cloud observation,
a short force history, a candidate action. Out: the next decision's per-segment force and peak
pressure. Supervised, MSE, on logged rollouts.

Three uses, and the second is the one FCVP could not have:
- **at execution**, as their filter;
- **during training**, as a shield on data collection, so exploration is bounded from the first step;
- **as an auxiliary target** for the policy's encoder, if the gating probe says force is learnable
  from the observation.

### 4. The policy trainer — one day [E] on top of what exists

Actor on the point cloud, deployable. Critic on the point-cloud history **and** the force summary,
training only. The critic must read both: conditioning it on the privileged signal alone biases the
policy gradient under partial observability. The asymmetric path already exists in `sac.py` and
`models.py`; the work is threading the force summary into the privileged vector and the replay.

### 5. The ablation that is the actual contribution — the same trainer, three signals

Identical architecture, identical budget, three sources of the training-time force:
1. **the exact per-vertex IPC force**, which only this simulator gives;
2. **a crude spring proxy**, which we already have for free: the cuff's hold is a spring of known
   stiffness and its displacement, the tracking error, is already in every info and in the privileged
   state [MI]. This is GentleHumanoid's choice, reproduced here;
3. **no force at all**, which is Wang's setting.

**If 1 and 2 tie, force fidelity does not matter**, our simulator's advantage evaporates for this
purpose, and that is a result worth publishing because the field currently assumes the opposite
without testing it. **If 1 beats 2, that is the contribution**, and it is the first measurement of
whether contact-force accuracy matters to policy learning.

The proxy costs nothing to produce, so the ablation is nearly free. Nobody has run it.

## The mechanism is settled by two independent results: condition on force, do not predict it

**FMVP re-tested FCVP's predict-and-filter and it lost.** From the paper, on the same task: "FCVP
predicts next-step forces and filters high-force actions, but this is less effective under arm motion,
where future arm positions are uncertain and the garment may be pulled in ways that induce
unanticipated force. By conditioning on force feedback and directly optimizing for dressing success,
FMVP achieves more robust performance." [RA] That is a controlled comparison by a third party, and it
says the filter is the weaker mechanism.

**RoboPack found the same shape on tactile.** Tactile as an input scored 16 of 20; no tactile 8 of 20;
and the variant that *predicted* future tactile scored **6 of 20, below the no-tactile ablation** [RA].

So the design decision is made for us: **force is a conditioning input to the policy and a signal the
critic reads, not a quantity the model is trained to predict.** Component 3 above is demoted from the
centre of the design to a training-time convenience, and the auxiliary force-prediction head is now
evidence-against rather than evidence-for.

## And the gap is named by the authors we would be competing with

FMVP, explaining why its simulation stage has no force: "this simulation training does not use any
force information, **as current simulators lack realistic force modeling for deformable garments**"
[RA]. It therefore conditions on force only in real-world fine-tuning, from real sensors, on 192
trials.

| Work | Where force enters | Where the force comes from |
|---|---|---|
| Wang RSS 2023 | a penalty term in the reward | a FleX proxy in arbitrary units |
| FCVP RA-L 2024 | an execution-time action filter | 264 real trajectories |
| FMVP CoRL 2025 | conditioning the policy, in real-world fine-tuning | real robot sensors, 192 trials |
| **online training in simulation** | **nobody** | **no simulator reports it** |

That last row is the opportunity, and it is narrow: not better physics, but the one arrangement the
field's own stated limitation rules out for everyone else.

## Three corrections from the sim-to-real evidence

**1. The deployable quantity is the gripper-side wrench, not the force on the arm.** FCVP thresholds
the Sawyer's own force/torque reading and Erickson's watchdog is a wrist sensor: both are the wrench
at the robot, which carries the garment's weight, its inertia and the robot's own friction. Training a
predictor on the arm's contact force and thresholding it against a wrist sensor would be wrong even
with a perfect contact model. **We already have the gripper-side quantity for free**: the cuff's hold
is a spring of known stiffness and its displacement, the tracking error, is in every info and in the
privileged state [MI]. So the design needs both channels, and they play different roles — the arm's
force is the privileged safety quantity, the gripper wrench is the deployable one.

**2. Fixing the units does not fix the values.** libuipc does fix a real problem: FCVP's simulated
threshold is stated in "units" with the note that they "do not correspond to Newtons in the real
world" [RA], and the same is said of PhysX cloth parameters. Our newtons are newtons. But the
dominant error is the unidentified cloth-skin friction coefficient, and six-digit verification only
means the solver solves *its own* contact model exactly. Measured on exactly our contact pair — inner
forearm against hospital fabric — the friction coefficient rises 26 to 43 per cent between dry and
normally moist skin and more than doubles against wet fabric [RA]. Every precedent that made a
sim-trained force model work did system identification first: Erickson fitted garment stretch,
stiffness, shear and friction by CMA-ES against real robot data before the transfer worked [RA].

**3. Simulation is wrong exactly where it is unique, and this is the strongest argument against the
whole direction.** Its only irreplaceable advantage is the dangerous high-force regime nobody may
ethically collect from people. But that is the regime it gets wrong: Yu et al. had to *exclude*
caught sequences from their parameter fit because "the rapid increase of force in caught sequences
lead to sub-optimal simulators with an unrealistically large friction coefficient" [RA], and FCVP
states that the garment catches which dominate real force "usually do not occur in simulation" [RA].
That is a shift in the labels, not noise: the snag begins at a different action, so a predictor learns
a systematically wrong decision boundary. Against that, FCVP needed only 264 real trajectories, about
a day of robot time, so simulation buys little on volume.

**What survives all three.** The three-way ablation in component 5 is a pure simulation experiment:
the policy is trained and scored in simulation, so whether our newtons match a real forearm does not
enter. It answers whether force fidelity matters to *learning*, which is answerable here and is
unanswered anywhere. What the corrections do bite is the claim that a sim-trained force model is
deployable, and that claim now needs the measurements below before it can be made.

**The validation nobody has done**, and the one that would settle it: replay recorded real
end-effector trajectories open-loop in the simulator and report the wrench error and correlation
against a real force/torque trace, per pose and garment. We cannot run it today — a Stretch 3 has no
force/torque sensor — so it needs either a borrowed sensor or a different robot.

## Order of work

1. **Phase 0, half a day**: gate the transients on Newton telemetry, and ship **both** force
   channels into `infos` — the arm's contact force and the gripper-side wrench from the hold's
   stiffness times its tracking error. Without the gate every number below fits the solver's opening
   iterates.
2. **The aggregator, one day**: per-segment resultants and peak pressure per square centimetre, on
   both channels.
3. **The three-way ablation, about three days of compute** [E]: the contribution, and pure simulation,
   so the sim-to-real corrections above do not touch it.
4. **The sim-trained force model, two days**, but only as a training-time shield until the material
   measurements below are done. Claiming it is deployable needs the garment's mass, thickness, stretch
   and bending stiffness measured, the cloth-skin friction coefficient measured on a forearm with the
   26 to 43 per cent hydration spread as the randomisation range, and the open-loop wrench validation.
   libuipc takes physical parameters directly, which is a real advantage over PhysX and FleX here.
5. Only then the larger architecture — a privileged world model, a curriculum over the continuous
   pose space — which is where the sample-efficiency gain would have to come from, and which has no
   published precedent on cloth.

## What would sink it

- **The accuracy ablation ties.** Then the simulator is not the differentiator and the honest move is
  to say so.
- **Simulated force does not resemble real force on a limb.** Our arm is a rigid mesh; a real limb is
  compliant. Deep Haptic MPC's transfer is the precedent for optimism, TaCauchy's agreement with real
  tactile responses at 1.3 to 4.7 N is the nearest validation [RA], and neither is a garment on a
  human arm.
- **The friction channel stays unusable** [MI], so everything here bounds normal pressure only, while
  shear is what injures frail skin.
