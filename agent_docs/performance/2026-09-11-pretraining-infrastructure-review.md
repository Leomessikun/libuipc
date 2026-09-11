# The pretraining infrastructure, reviewed from first principles

Wall-clock time to a trained policy is the number of transitions a recipe needs, times the seconds each
transition costs, divided by the number of transitions produced in parallel. Wang's recipe (point-cloud
SAC, regional teachers distilled into a student) fixed the first factor for PyFleX. This port swapped
the simulator for libuipc IPC and kept the recipe. Every issue below is a term in that product. Figures
come from relaunch 3 of the region-13 teacher (`wang_teacher_r13_s1`) unless stated otherwise;
unmeasured claims say so.

## The budget is below Wang's own numbers

- Relaunch 3 produces 11.6k transitions per hour on one RTX PRO 6000 Blackwell with 24 environments,
  so a 600k teacher takes 2.2 days and the chain of three teachers and a student about nine.
- Wang's teachers reached their best checkpoints between 0.52M and 1.96M transitions, median 0.75M.
  The chain's 600k is below that median. At this rate the median is 2.7 days per teacher, the top of
  the range seven, and Wang's 23.5M teacher total 84 days.
- Faithful execution is therefore under-budgeted by construction. The 300k checkpoint is where the
  budget is either raised or traded for a deviation that needs fewer transitions.

## Where a transition's time goes

| Part | Share of wall time |
|---|---:|
| Simulation (IPC step, observation, reward) | 68% |
| Evaluation | 18% |
| Network updates | 15% |
| World rebuilds | 1% |

- **The parts run one after another** in a single process on a single GPU. Any other GPU job slows
  training directly: the AL-release probes of the same day took the teacher from 3.8 s to between 6
  and 9 s per vector step.
- **Evaluation follows Wang's protocol.** It runs one deterministic episode for each of the 25
  held-out configurations every 10k transitions, as `curl/train.py` does. Cutting it is a deviation.
  Running it on another process or GPU against saved checkpoints keeps the protocol and takes 18% off
  the critical path.
- **The teachers run in series** although they are independent. Three GPUs would cut the teacher phase
  to a third without any algorithmic change. This is the largest lever.
- **Copies per world.** Going from 16 to 32 copies raised throughput from 6.8 to 10.5 transitions per
  second ([dressing correctness](2026-09-09-dressing-correctness.md)). The teacher runs 24 copies in
  29 of 96 GB. The gain from 24 to 32 has not been measured.
- **Overlap.** Overlapping updates with simulation is capped at 15%. Whether IPC leaves the GPU
  headroom for it has not been measured.

## Fast pretraining in Newton, fine-tuning in IPC, is not worth it

The idea was to spend most transitions in a cheaper simulator (Newton's VBD cloth) and fine-tune in IPC.
The measurements do not support it:
- **The speed gap is mostly a different MDP.** The quoted 4.4 times compared Newton at one simulation
  step per decision with this port at six. At six steps per decision, Newton's 40 environments give
  11.5 transitions per second against 10.5 for this port's 32 copies: 1.10 times. At 16 copies the gap
  is 1.70 times ([dressing correctness](2026-09-09-dressing-correctness.md)).
- **VBD differs exactly where the task is hard.** VBD cloth stretches and penetrates slightly, so the
  scripted expert reaches the 0.7 threshold in 20 of 35 cells there. It also collapses a closed cuff,
  which is why Newton needs pre-baked states ([pretraining](2026-09-08-uipc-manip-pretraining.md)).
  A policy that learns to get past the elbow by stretching the sleeve does not transfer.
- **Newton's own SAC did not solve the task.** Its 33 evaluated runs peak at one success in 40
  episodes after 2.98M transitions.
- **The cost is two simulators.** Garments, bodies, observations, actions and rewards must match
  exactly, and the critic's values are simulator-specific.

Sim-to-sim pretraining pays when the cheap simulator is an order of magnitude faster and agrees on the
task-critical contact. Here it is 1.1 to 1.7 times faster and disagrees at the elbow.

## The grasp is the structural weak point

Wang's PyFleX picker attaches particles kinematically. This port holds 48 anchors with a soft position
constraint and bounds them with the 6 cm vertex tether ([solver stall](2026-09-11-dressing-solver-stall.md)).
- **Before the tether,** held vertices trailed their targets by up to 179 mm. This caused the slow
  tail that stalled relaunches 1 and 2.
- **The tether drops a whole move** whenever a held vertex would trail by more than 6 cm. The policy
  cannot observe why its action had no effect. How much that costs learning has not been measured;
  counting dropped moves per episode would measure it.
- **The watchdog.** By 84k transitions it tripped three times, at episode steps 225 to 279. The last
  two trips ran with no other GPU job. Each trip drops 24 transitions from the slowest,
  contact-hardest states. The rate is about one per 13k; at more than about one per 10k the replay
  would under-sample the elbow.
- **The real fix is not available.** It is the penalty-free moving boundary of AL-IPC (Zheng, Luo and
  Li, section 5.3), which no libuipc version implements. This is a cost of choosing IPC, not a bug.

## Sample efficiency: what the recipe leaves unused

- **The expert is not used.** The scripted expert dresses 22 of 40 cells in this port with the tether.
  Wang's agent is `SAC_AWAC`, and his launcher has a `demo_buffer_dir` option, set to `None` in the
  final configuration. Seeding the replay with expert episodes targets the first few hundred thousand
  transitions, where Newton's working runs showed no upper-arm progress through 280k. Not measured
  here.
- **The teachers need not see point clouds.** They exist only to be distilled. The asymmetric critic
  on a 35-float privileged state is built and makes updates 2.1 times faster (2.85 times in bf16).
  Its effect on sample efficiency has not been measured in this port.

## Contact forces are available but not wired in

- **Arm contact force.** The 0.0.28 wheel exposes `uipc.core.ContactSystemFeature`, with contact
  energy, gradient and Hessian per primitive type: point-triangle, point-edge, point-point, edge-edge
  and point-half-plane, each split into normal (`+N`) and friction (`+F`). The force on an arm vertex
  is the negated sum of its gradients.
  - The gradient is in the solver's incremental-potential units. Converting it to newtons, probably by
    dividing by dt squared, needs a calibration.
  - Calibration test: a sleeve resting on the arm must press on it with its own weight.
  - Wang's PyFleX signal was the contact multipliers λ, which depend on iteration count and step size.
    His force-penalty threshold of 500 is empirical. A barrier gradient on a penetration-free state
    is the cleaner signal.
  - This port's reward has no force term (`dressing_reward.py`). Wang's penalises force above 500
    with weight 0.001.
- **Grasp pull force.** It is the hold spring times the hold gap. This needs no API, but its size is
  set by the chosen constraint strength.
- **Robot coupling force.** Genesis's `two_way_soft_constraint` coupling feeds a robot link
  `translation_strength * mass * (x_ipc - x_aim)` (`ipc_coupler/utils.py`). That is a spring
  reading, a step behind and stiffness-dependent. It is usable as a coarse wrist-force signal once the
  Stretch 3 is in the loop.

## What is not the lever

- **The solver.** IPC is not time-of-impact locked on this scene: it takes 4 to 6 Newton iterations
  per frame. AL-IPC is 2.2 times slower here, and the authors' release adds nothing upstream lacks
  ([solver stall](2026-09-11-dressing-solver-stall.md)). Contact parameters and step size were not
  levers.
- **The SAC hyperparameters match Wang's launcher** (`curl/launch_train_curl.py`): batch 64
  (`get_batch_size` for `pointcloud_3`), an actor update every fourth critic update, learning rates
  1e-4, and evaluation every 10k. Neither side clips gradients. Time limits bootstrap on both sides.
  The one difference is the temperature learning rate, 5e-5 here against 1e-4.
