# Our critic is the architecture the reference measured and rejected

Date: 2026-09-12. Verified against the reference's own paper and our own code.

## What the reference does, and what it rejects

Wang et al., RSS 2023, describe their Q function verbatim:

> "given the current point cloud P and the action sampled from the Dense Transformation policy
> a* ~ π(P), we **concatenate a\* as an additional feature to every point p_i in the input point
> cloud**, so the feature of each point includes its 3D position, the one-hot vector of the class
> type, and the action a*. We then use a classification-type neural network architecture (i.e.,
> classification-type PointNet++) to output a scalar Q value."

And they name the alternative as a baseline:

> "**Dense Transformation policy + latent Q function**: This baseline uses the same policy as our
> method. For the Q function, instead of concatenating the action to the input point cloud as an
> additional feature, this baseline first uses a PointNet++ to encode the point cloud observation
> into a latent vector, and then concatenates the action to the latent vector."

Their Table I, upper-arm dressed ratio on three pose sub-ranges, three seeds [RA]:

| Sub-range | Dense Transform (theirs) | **Dense Transform, latent Q** | Direct Vector | TD-MPC | Deep Haptic MPC |
|---|---:|---:|---:|---:|---:|
| 1 | 0.68 ± 0.05 | **0.55 ± 0.05** | 0.63 ± 0.07 | 0.04 ± 0.04 | 0.31 ± 0.08 |
| 2 | 0.55 ± 0.06 | **0.42 ± 0.08** | 0.52 ± 0.09 | 0.00 ± 0.00 | 0.11 ± 0.02 |
| 3 | 0.75 ± 0.04 | **0.68 ± 0.07** | 0.73 ± 0.02 | 0.03 ± 0.05 | 0.37 ± 0.02 |
| Mean | 0.66 | **0.55** | 0.63 | 0.02 | 0.26 |

The text summarises it: "Using a 'Latent Q' architecture leads to much worse performance."

## What we implemented

`models.py`, `Critic`:

```python
"""Twin Q critic in the reference form ``Q(encode(s), a)``: the action joins after encoding."""
...
z = self.encoder(pos, feat, valid)
parts = [z, action]
z = torch.cat(parts, dim=-1)
return self.Q1(z), self.Q2(z)
```

The encoder is a **global** PointNet++ reducing the cloud to 50 numbers, and the action is
concatenated to that latent vector. **This is the latent-Q baseline, not the reference's method**, and
the docstring calls it "the reference form", which is wrong.

The actor looks right: `WangFlowActor` reads the tool point's row out of a segmentation encoder,
which is the Dense Transformation shape.

## What it is worth, and what it does not explain

In the reference's own measurements the latent-Q choice costs **about 0.11** of upper-arm dressed
ratio, averaged over their three sub-ranges [RA].

**It does not explain our gap on its own.** We train one region, which is a single sub-range, so the
comparison is with 0.55 for the latent-Q column. Our best honest evaluation is **0.283**. The
architecture defect accounts for roughly 0.11 of a 0.38 shortfall against the reference's best and
leaves about 0.27 unexplained against the same architecture they measured.

## Why this was not visible before

The port's records compare our numbers against 0.740, which is the reference's **region-13 teacher
checkpoint** from their released files, and against 0.68, which is their **distilled student over all
27 regions**. Neither is the right comparison for a single-region policy: Table I is, and it was not
consulted. On Table I's scale a flat policy over the *whole* pose range scores 0.34 and ties their
heuristic planner at 0.32 (Table II), so the ambition of a single flat policy was never 0.68.

## What to do

Implement the reference's Q function: concatenate the action as a per-point feature and run a
classification-type PointNet++ over the augmented cloud. It is a change to `Critic.forward` and the
encoder's input width, not to the training loop. Then re-measure on the same held-out region-13
configurations against 0.283, with the seeds a comparison needs.
