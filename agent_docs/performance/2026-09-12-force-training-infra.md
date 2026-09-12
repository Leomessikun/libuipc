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

## Order of work

1. **Phase 0, half a day**: gate the transients on Newton telemetry, ship the force summary into
   `infos`. Without this every number below fits the solver's opening iterates.
2. **The aggregator and the sim-trained force model, two to three days**: this alone reproduces what
   FCVP bought with 264 real trials, and is a deliverable on its own.
3. **The three-way ablation, about three days of compute** [E]: the contribution.
4. Only then the larger architecture — a privileged world model, a curriculum over the continuous
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
