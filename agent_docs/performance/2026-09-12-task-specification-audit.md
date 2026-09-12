# Is our task the same task? A comparison with the reference

Date: 2026-09-12. Status: partial, two explanations ruled out, one measurement running.

Before choosing a new way to train the policy, the least interesting explanation for six months of
failure has to be ruled out: that our port specifies a harder task than the one the reference solved,
so that no method would fix it. The scripted controller scores 0.542 on held-out poses where our SAC
teacher reached 0.283, and the reference reports 0.740 for this region's teacher.

Tags: [MI] measured here, [RA] read from the reference's code, [E] inferred.

## Ruled out

**Travel budget. Ours is three times the reference's.**

| | Ours | Reference |
|---|---|---|
| Per decision | 8.66 mm per axis, 15.0 mm resultant [MI] | 1 cm per step [E] |
| Decisions | 300 | 150 (`launch_train_curl.py:241`) |
| Budget over an episode | **4.50 m** | 1.50 m |

Our per-axis cap is the 0.15 m/s end-effector speed over a 0.1 s decision divided by the square root
of three (`dressing_env.py`, `max_translation`). The reference's per-step displacement is not stated
directly — `PickerMultipleParticle` is defined in the authors' own softgym fork, which is not in the
checkout — but its own code implies it: `translation_move_steps = int(translation_distance /
self.max_translation * 10)` (`dress_env.py:1328`) with `max_translation=0.1`
(`launch_train_curl.py:284`), so a step covers a tenth of `max_translation`, one centimetre. Marked
inferred rather than read.

**The reward. Same shape, same weights.** The reference sums
`task_w * task + force_w * force + collision_w * collision + alignment_w * alignment + ...` and
clamps at −100 (`dress_env.py`, `compute_reward`); its task term is the negative distance to the
garment's shoulder opening off the arm, and the forearm length plus `upper_w` times the upper-arm
distance on it, with `upper_w = 5` (`:52`). Ours is the same function with the same upper-arm weight
of five and the same −100 clamp (`dressing_reward.py:141,166-168`). The one term we lack is the force
penalty, `force_w = 0.001` above a threshold of 500 (`dress_env.py:49,53`), which is a FleX contact
proxy in arbitrary units.

## The difference that remains: the grasp

The reference holds the garment with a picker that carries its particles kinematically. We hold 48
cuff vertices with a soft position constraint of strength 1e4 and add two rules the reference does
not have in the same form:

- **A 6 cm tether**: a sub-step that would leave any held vertex further than `anchor_tether_m` from
  its commanded position is discarded whole, rotation included (`dressing_env.py`, `tether_allows`).
- **A 12 mm no-move rule**: a sub-step that would put the anchor within
  `no_move_collision_threshold` of the arm keeps the rotation but drops the translation. The
  reference has this one too (`no_move_collision_threshold=0.012`, `launch_train_curl.py:69-70`).

The hypothesis was that an exploring policy trips the tether where the smooth scripted controller
does not, so the learner acts in an environment that silently voids some of its actions and cannot
see that it did. **It is wrong.** Counting every call inside the sub-step loop, 4,800 per policy over
200 decisions on four cells [MI]:

| Policy | Tether rejections | Held-vertex gap, median | 90th percentile | Maximum |
|---|---:|---:|---:|---:|
| Scripted controller | **0.00 %** | 4.40 mm | 19.17 mm | 26.35 mm |
| Uniform random actions | **0.00 %** | 2.10 mm | 2.82 mm | 8.13 mm |
| Controller plus Gaussian noise | **0.00 %** | 4.30 mm | 30.06 mm | **47.72 mm** |

Against a 60 mm tether, the worst case over 4,800 sub-steps of the noisiest policy reaches 48 mm and
never trips. Random actions load the cloth *least*, because they wander away from the arm rather than
pulling against it. The only rejections anywhere come from the 12 mm no-move rule, 10.2 per cent for
the noisy controller and zero for the other two — and the reference has that rule too.

**So the grasp does not explain the gap either.**

## What this leaves

No difference in the task's specification accounts for a hand-written controller beating a learned
policy here. The travel budget is three times the reference's, the reward is the same function with
the same weights, and the two grasp rules that are ours alone do not fire. What remains is the
simulator's own dynamics, which are a penetration-free implicit solve rather than position-based
particles, and the method itself.

## Not yet compared

The observation and its camera model, the episode boundary and bootstrapping, and whether the
(garment, pose) distribution our teacher draws from matches the reference's region 13. These need the
same treatment.
