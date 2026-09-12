# Putting contact force into training, and what it replaces

Date: 2026-09-12. Status: design under one gating experiment, which is running.

The question this answers is not what the force looks like. It is how force enters policy
training, and whether it supports an architecture that replaces Wang RSS 2023's rather than
decorating it. Six months reproducing that architecture in this simulator produced no progress, and
both roads its authors' line has taken since — the regional teachers, and demonstrations plus a
diffusion policy — are published by other groups.

Tags: [RA] reported by the authors, [MI] measured here, [E] our estimate.

## Two findings that remove options before any design

**A force penalty in the reward is the published baseline, not a contribution.** Wang's own
environment carries it: `force_w=0.001`, "weight for force penalty, when the applied force is too
large on the human", above `force_threshold=500` (`dress_env.py:49-53`). His reward is
simulator-privileged throughout. So the obvious first idea — add a force term to the reward — is
already what the field does; only the fidelity of the number would be new, and a number nobody can
compare is not a result.

**Wang's stated reason for the 27 teachers is not supported by his own ablation.** He writes that a
single policy is hard to learn because of "imbalanced learning speed for different tasks" and
"conflicting gradients from different tasks". PCGrad is the direct remedy for conflicting gradients.
His Table II [RA]:

| Method | Upper-arm dressed ratio |
|---|---:|
| Policy distillation, his method | 0.68 ± 0.012 |
| Policy distillation, KL | 0.45 ± 0.010 |
| **PCGrad** | **0.37 ± 0.063** |
| **No distillation, one policy** | **0.34 ± 0.10** |
| Heuristic motion planning | 0.32 |

PCGrad moves 0.34 to 0.37 with error bars of 0.063 and 0.10. It does nothing. So the gradient-conflict
half of the diagnosis fails on his own evidence, and the multi-task-optimisation literature agrees
that these methods do not beat a tuned weighted sum. What survives is the other half, imbalanced
learning speed, which is a credit-assignment problem: in a hard pose the reward stays flat until the
garment clears the elbow, so the hard poses get almost no learning signal for most of training.

**Contact force observes exactly that delay**, densely, per step, on failing episodes as well as
succeeding ones. That is the reason to expect it to matter, and it is different from the reason the
field usually gives, which is safety.

## The gating experiment, which decides everything below

Both independent surveys converged on the same test, from the failure analysis of privileged-learning
methods: a method that regresses a privileged signal from the deployable observation only works when
the observation predicts the signal.

> Train a probe from point-cloud observations to the binned contact force on the arm, offline, on
> logged episodes, and report the coefficient of determination.

- **High R^2**: the segmented point cloud already determines the contact state. Force is not hidden
  state, and every mechanism below that supervises a representation with force is redundant.
- **Near-zero R^2**: force is hidden state, which is the interesting case, but then every mechanism
  that *regresses* force from the observation is ill-posed by construction. Only mechanisms where
  force enters the critic, the dynamics model or the exploration survive.
- **In between**: the auxiliary-head family is worth building.

Either answer eliminates a family, which is why it runs first. It costs one collection run and an
offline fit, no training.

## The gating experiment was run, and it is confounded

1,200 paired samples were collected: the scripted expert on four held-out region-13 cells, one
environment each, 300 decisions, saving the deployable observation beside the contact force on the
arm binned into eight bands along the fingertip-to-shoulder axis plus its total and peak. A network
was then fitted from observation to log1p force.

| Split | Test R^2, total force | Test R^2, peak | Test R^2, bands |
|---|---:|---:|---:|
| Random 75/25, cells mixed | +0.935 | +0.889 | +0.951 |
| Held out hospital_gown on 14047 | -0.752 | -0.214 | -0.795 |
| Held out tshirt_26 on 14045 | +0.264 | +0.220 | +0.745 |
| Held out tshirt_4 on 14046 | -1.055 | -0.830 | -0.087 |
| Held out tshirt_68 on 14048 | -0.024 | -0.040 | +0.507 |
| **Control: the tool's 7 numbers alone, random split** | **+0.958** | **+0.916** | **+0.943** |

**The control invalidates the random split.** Predicting the force from the gripper's own position and
goal offset, seven numbers with no point cloud at all, scores *higher* than the full 5,383-dimensional
observation. The expert follows a nearly deterministic path, so the tool's position stands in for the
phase of the episode, and the force is a function of phase. A random split over timesteps of the same
trajectory therefore leaks: neighbouring rows are almost the same state. The +0.935 measures
autocorrelation, not perception.

**And the leave-one-cell-out split is under-powered.** Three training cells cannot teach a model to
generalise to a fourth arm geometry and garment, so R^2 near zero there is what any signal would give.

**So the experiment as both surveys specified it does not decide the question.** A valid version needs:
- many cells, tens rather than three, so leave-one-cell-out has a population to generalise over;
- a split by cell and by episode, never by timestep;
- the tool-only control reported beside every number, since it is the thing to beat;
- policies other than the one scripted expert, whose determinism is what creates the confound;
- and the transient gate below, because the target is currently junk-dominated.

**The target is transient-dominated.** Over these 1,200 samples the total force on the arm has a
median of 87.6 N but a mean of 352 N, a 95th percentile of 2,070 N and a maximum of 2,938 N. Fitting
that is fitting the solver's opening iterates. Gating on the engine's per-frame Newton telemetry has
to come first, and it is untested.

This is the cheap experiment doing its job: it cost one collection run and it stopped a design from
being built on a number that measures the wrong thing.

## What survives either answer

**Force in the critic, not in the actor.** The actor reads the point cloud; the critic reads the
point-cloud history *and* a low-dimensional force summary. The critic must take both: conditioning a
critic on the privileged signal alone gives a biased policy gradient in a partially observed problem,
which is the standard error in asymmetric actor-critic. A state-dependent privileged signal, which
contact force is, leaves the gradient unbiased. Our trainer already has the asymmetric path, and the
force summary is a handful of scalars.
- Its own test, before any policy is trained: does the force-conditioned critic predict returns on
  held-out episodes better than the force-free one? If not, force carries no return-relevant
  information and the rest collapses.

**Force in a world model.** A model-based learner trained on both observations, with the force in the
dynamics, the reward predictor and the critic, and the policy acting on the deployable observation
only. This is the shape both surveys ranked first, on the evidence that it beats the
distillation-family baselines on their own benchmark [RA]. It also decouples gradient steps from the
simulator, which is the only real lever at 11.6k transitions per hour [MI]. No published evidence on
deformables.

**Force as the curriculum's measure, over the continuous pose space rather than 27 bins.** A sampler
that spends time where learning progress is highest needs a progress measure defined on failing
episodes, where return is flat. Arm length dressed at the first sustained contact, contact count and
peak settled force are all such measures. This addresses the half of Wang's diagnosis that survives,
costs no gradient steps, and composes with the above.

## What our own code makes cheap, and what it forbids

From a read-only audit of the trainer:

| Insertion point | Work [E] | What it breaks |
|---|---|---|
| Force summary into `infos` | half a day | nothing; prerequisite for choosing any scale |
| Force into the privileged critic | half a day | `PRIVILEGED_DIM` changes, so saved replays and privileged checkpoints are rejected |
| Force term in the reward | a day | return comparability; the reward scale and the critic's target clamp were sized for metre-scale rewards |
| Auxiliary force head on the actor's features | two days | replay signature; update cost barely moves, the features are already computed |

Structurally forbidden without a rewrite:
- **Force as an actor input.** The observation width is baked into the checkpoint protocol; every
  checkpoint and replay becomes unloadable.
- **Per-point force targets on the observed cloud.** The observation pipeline discards the map from a
  sampled point back to a cloth vertex, so the natural per-point auxiliary task needs a new
  observation path.
- **Termination on a force threshold.** All slots share one world and one horizon, and the protocol
  raises if slots disagree. This removes force-triggered termination and the safe-RL formulations
  that depend on it.

## The measurement facts any of this has to respect

- Force in newtons is the exported contact gradient over dt squared, exact to six digits in three
  scenes [MI].
- **A settled force reproduces to 3 to 5 per cent between identical runs; a transient one differs by
  a factor of nine** [MI]. So the signal must be a settled or aggregated quantity. A reward or an
  intrinsic reward on the instantaneous peak would fit noise.
- **Transients reach three orders of magnitude** above the physical value in the opening frames of a
  contact [MI], because the gradient is the last Newton iterate. Every mechanism here consumes force
  magnitude, so this has to be gated before any of them. The engine exposes per-frame Newton
  telemetry — iteration count, line-search trials, convergence flags — which is the obvious gate and
  is untested.
- **The friction channel is not usable as exported** [MI]. None of the mechanisms above need it.
- **A force the robot could actually feel already exists in the loop**: the cuff's hold is a spring of
  known stiffness and its displacement, the tracking error, is already in every info and in the
  privileged state. It is the only force a Stretch 3 can sense, through joint effort.

## The plan

Four phases. The first three are cheap and each can kill the direction; only the fourth is a build.
Nothing here reproduces Wang's architecture or clones demonstrations into a diffusion policy.

### Phase 0 — make the force number mean something. Half a day, one GPU hour.

Without this every later number fits the solver's opening iterates: the arm's total force has a
median of 87.6 N and a maximum of 2,938 N over the same 1,200 samples.

1. Log `Engine.frame_stats()` beside the force: Newton iterations, line-search trials, convergence
   and limit flags, CCD time of impact.
2. Test the gate: does a force spike coincide with a frame that did not converge, or that hit the
   Newton cap? If it does, the gate is a flag and the signal becomes usable. If it does not, gate on
   persistence instead: report a contact only once it has held across consecutive settled reads.
3. Ship the force summary into `infos` (insertion point A from the code audit, breaks nothing).

**Kills the direction if:** spikes track nothing the engine reports and do not settle. Then the force
is not measurable per step at production tolerances, and only episode-level statistics survive.

### Phase 1 — repeat the gating probe properly. One to two days, six GPU hours.

The probe already run is confounded: the gripper's own seven numbers beat the whole point cloud,
because the expert's path makes tool position a stand-in for episode phase.

1. Collect from tens of cells, not four, across all five garments, with a stochastic policy as well
   as the expert so the trajectory is not deterministic.
2. Split by cell and by episode, never by timestep. Report the tool-only control beside every number.
3. Fit observation history to gated force.

**Decides the architecture:** high R^2 means force is not hidden state and the representation
mechanisms are redundant; near zero means they are ill-posed and only force in the critic, the
dynamics and the exploration survive; in between means the auxiliary head is worth building.

### Phase 2 — the cheapest test that force matters at all. Half a day of code, nine GPU hours.

Independent of Phase 1's answer. Train two critics on the same logged episodes, one reading the
point-cloud history and one reading that plus the gated force summary, and compare value-prediction
error on held-out episodes. The critic must read both, never the privileged signal alone, or the
policy gradient is biased.

**Kills the direction if:** the force-conditioned critic is not materially better. Then force carries
no return-relevant information here and no architecture built on it will help.

### Phase 3 — the build, only if Phases 1 and 2 pass. One to two weeks.

Two independent surveys converged on the same shape:

- **A privileged world model.** Dynamics, reward predictor and critic see the gated force; the policy
  acts on the deployable observation alone. No imitation objective between them, so this is not
  distillation. It also decouples gradient steps from the simulator, which is the only real lever at
  11.6k transitions per hour.
- **A learning-progress curriculum over the continuous pose space**, replacing the 27 hand-cut
  regions. Its progress measure comes from force: arm length dressed at the first sustained contact,
  sustained-contact count, peak settled pressure. These are defined on failing episodes, where return
  is flat, which is the half of Wang's diagnosis that survives his own ablation.
- **A two-day control:** one policy on a learned pose embedding with tuned loss scalarisation. If the
  worst poses sit at Wang's single-policy 0.34, weighting was never the problem, and a whole branch
  is eliminated cheaply.

### What the contribution would be

Not a force penalty, which is Wang's published baseline and which Clegg measured at zero task success.
The claim is the quantity: **peak local pressure over roughly one square centimetre of skin, per body
segment**, which per-vertex contact force can compute and a wrist force reading cannot. The field has
no shared definition — two groups quote 18 N and 120 N for different things, real dressing work uses
10 N, 1.02 N and 5.4 N with no standard cited, and ISO/TS 15066's own table is force *and* pressure
per body region from 100 healthy 18-to-66-year-olds, which is an extrapolation to a frail arm.

### The honest gap

Shear is what tears frail skin, and the friction channel is unusable as exported. Everything above
bounds normal pressure only. Fixing the friction readout is a separate piece of work and its own
possible contribution.
