# What the existing dressing experience contains — 2026-09-17

Status: offline measurement of saved replays and logs, CPU only; nothing
trained or simulated. It tests the data premise of the
[shared FQL pretraining design](2026-09-17-rl-pretraining-contract.md)
("existing experience can reduce the simulation needed to learn") and answers
the owner's question whether to continue the RL idea. Script and numbers:
`output/uipc_manip/sleeve_path_audit_20260917/prior_data_inventory.py` and
`.json`.

## Compatible replays are almost entirely failure

Eight saved replays share the current contract (5,383-D observation, 6-D
action, 24 slots, horizon 300, five garments). A slot-episode "reaches the
zone" if its reward, divided by the reward scale, ever exceeds 1.4 (forearm
length plus five times 0.7 of the upper arm; approximate, it varies by body).

| Run | Transitions | Slot-episodes | Reach the zone | Decisions in the zone | Best / last evaluation of 25 |
|---|---:|---:|---:|---:|---:|
| `wang_teacher_r13_s1` | 265,584 | 864 | 59 | 2,500 | 3 / 0 |
| `abl_both_s1` | 132,216 | 432 | 40 | 1,942 | 7 / 2 |
| `abl_dense_s1` | 125,016 | 408 | 3 | 74 | 1 / 0 |
| `abl_base_s1` | 125,016 | 408 | 0 | 0 | 0 / 0 |
| `abl_residual_s1` | 125,016 | 408 | 0 | 0 | 0 / 0 |
| `abl_both_s2` | 125,016 | 408 | 0 | 0 | 0 / 0 |
| `pg_phys_s1` | 24,000 | 72 | 2 | 28 | 6 / 6 |
| `pg_ctrl_s1` | 24,000 | 72 | 1 | 67 | 0 / 0 |
| **Total** | **945,864** | **3,072** | **105 (3.4 %)** | **4,611 (0.49 %)** | |

Every one has `priv_dim` 0. The two `dressing_redesign_20260916` replays are
the 125,016 rows of `abl_dense_s1` plus 2,400 and are not counted twice.

On the same 25 evaluation cells, the scripted expert
(`expert_r13_heldout_s0`, 733 s of simulation, no training) scores 11 by the
final reading and 15 at the peak. The best reinforcement-learning evaluation
ever logged on those cells is 7 (one snapshot of `abl_both_s1` at 110k
transitions, 2 at its end); `wang_teacher_r13_s1` spent 58,238 s in the
environment and 24,080 s in evaluation for 265k transitions and ends at 0.

## Consequences for the FQL design

1. **The behavior prior would learn the failing behavior.** A flow model fit
   to this corpus reproduces SAC policies that stall at the elbow; 0.49 % of
   its decisions are near success. Constraining the actor to that prior
   constrains it to failure. The successful behavior on disk is the scripted
   expert's 25 episodes and the 12 reconstructed ones, about 11,000 rows.
2. **The reward fix and the replays exclude each other.** The design requires
   a corrected reward and forbids mixing reward versions. The replays carry no
   geometry, so the shoulder-overshoot correction cannot be computed for them.
   After the fix, corrected-reward Bellman data is the ~11,000 expert rows;
   before it, the 0.95 M rows keep a reward whose branch switches carry 91 % of
   the squared one-decision change. A partial way out that needs no geometry:
   on-arm → off-arm drops are identifiable from the reward alone (2,000 rows,
   0.2 %) and can be masked from Q updates, at the cost of also masking
   genuine slip-offs. This fork is open in the design and must be decided
   before any learner is written.
3. **The comparison the design proposes is still the right one, on other
   data.** FQL, RLPD and plain imitation at matched total workstation time are
   a sound test of "does prior experience reduce simulation". It becomes
   informative once the prior contains success at a useful rate, which a
   repaired teacher can generate in hours (25 episodes per 733 s).

## Answer recorded for the owner

Reinforcement learning from scratch and with IPC labels has lost, on this
task, to a 733-second scripted controller. Reinforcement learning as
*improvement over a working teacher* is untested here and reasonable, and it is
where FQL-class methods belong. The order that follows from the measurements:
fix the success rule; repair and test the teacher on the 25 cells (the elbow
stall of the [sleeve-path audit](2026-09-17-sleeve-path-audit.md)); generate
geometry-logged data with it; then compare learners on that data. The choice
of learner is the last decision, not the first.

## Open problems in robot-assisted dressing (sourced)

1. **Moving arms in simulation.** Force-Modulated Visual Policy (CoRL 2025,
   arXiv 2509.12741, §5.2, verified): "cloth simulation with actuated humans
   in NVIDIA FleX is unstable and limited in fidelity", so its simulation
   training uses static arms and arm motion is learned from real data. Our
   environment's arm is a fixed rigid mesh as well. Stable contact against
   moving bodies is what IPC guarantees; this is an environment capability,
   not a learner.
2. **Force in simulation.** Same section: "current simulators do not offer
   accurate enough force modeling for deformable objects in contact with human
   limbs".
3. **Transfer across contact dynamics** (friction, stiffness, simulator) has
   no controlled test ([prior-art check](2026-09-17-sleeve-goal-prior-art.md)).
4. **No dressing work uses IPC contact** (same record).
5. **Training cost.** The reference pipeline trains 27 region-wise SAC teachers
   before distillation; the owner's single-workstation goal addresses this.
6. **Tight sleeves at the elbow.** Even the privileged scripted expert fails
   tshirt_392 on all five bodies.
