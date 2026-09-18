"""Invert the behavior flow: which noise would have produced this action?

The dressing behavior model is a flow-matching policy integrated forward with a fixed
number of Euler steps, so an action can be carried back along the same discretization
to the noise that generates it. That answers a question this project has only been
able to guess at: whether an action a teacher takes is something the learned prior can
already produce, and from where in its noise distribution.

Three quantities come out of one reversal, and they separate three different reasons a
prior might fail a teacher:

* the **latent norm** against the prior's own standard normal, which says whether the
  action sits in the bulk of the prior or in its tail;
* the **reconstruction error** after integrating the recovered noise forward again,
  which says whether the prior can represent the action at all;
* and, with a simulator, the **neighbourhood** of that noise, which says whether the
  behaviour is a basin an optimiser could find or a needle it could not.

The reversal is the explicit reverse of the forward Euler scheme, not an exact
inverse: the forward step evaluates the field before the step and the reverse step
evaluates it after, so the two differ by the integration error. That difference is the
point rather than a defect — it is what pulls a reference action toward what the prior
actually does — but it means the reconstruction error is never exactly zero and must
be read against the scale below.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def reverse_flow(agent, encoded: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Carry ``actions`` back through the behavior flow to the noise that produces them."""
    steps = int(agent.cfg.flow_steps)
    x = torch.as_tensor(actions, dtype=torch.float32, device=agent.device).clone()
    for k in reversed(range(steps)):
        time = x.new_full((len(x), 1), k / steps)
        x = x - agent.behavior.vector(encoded, x, time) / steps
    return x


@torch.no_grad()
def forward_flow(agent, encoded: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Integrate ``noise`` forward through the behavior flow, as the policy does."""
    return agent.flow_actions(encoded, torch.as_tensor(noise, dtype=torch.float32, device=agent.device))


@torch.no_grad()
def encode(agent, observations) -> torch.Tensor:
    """The behavior model's own encoding of a batch of packed observations."""
    flat = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=agent.device)
    return agent.behavior._frame_latent(agent.unpack(flat))


def chi_percentile(norm: np.ndarray, dimension: int, samples: int = 200_000, seed: int = 0) -> np.ndarray:
    """Where ``norm`` falls in the distribution of a standard normal vector's length."""
    rng = np.random.default_rng(seed)
    reference = np.sort(np.linalg.norm(rng.standard_normal((samples, dimension)), axis=1))
    return np.searchsorted(reference, np.asarray(norm, dtype=np.float64)) / samples


def reversal_report(agent, observations, actions, *, batch: int = 256) -> dict:
    """Latent norms, percentiles and reconstruction errors for a set of (observation, action) pairs.

    ``round_trip_noise`` is the reconstruction of the *noise* after a forward and a
    second reverse pass, which measures the integration error on its own, with no
    reference action involved, and so gives the scale the action error is read against.
    """
    observations = np.asarray(observations)
    actions = np.asarray(actions, dtype=np.float32)
    if len(observations) != len(actions):
        raise ValueError("Need one action per observation")
    latents, reconstructions, noise_round_trip = [], [], []
    for start in range(0, len(observations), batch):
        stop = start + batch
        encoded = encode(agent, observations[start:stop])
        reference = torch.as_tensor(actions[start:stop], device=agent.device)
        z = reverse_flow(agent, encoded, reference)
        forward = forward_flow(agent, encoded, z)
        again = reverse_flow(agent, encoded, forward)
        latents.append(z.cpu().numpy())
        reconstructions.append((forward - reference).cpu().numpy())
        noise_round_trip.append((again - z).cpu().numpy())
    latent = np.concatenate(latents)
    error = np.concatenate(reconstructions)
    drift = np.concatenate(noise_round_trip)
    norm = np.linalg.norm(latent, axis=1)
    return dict(latent=latent, latent_norm=norm,
                latent_percentile=chi_percentile(norm, latent.shape[1]),
                reconstruction_error=np.linalg.norm(error, axis=1),
                noise_round_trip=np.linalg.norm(drift, axis=1),
                action_norm=np.linalg.norm(actions, axis=1))


def summarize(report: dict) -> dict:
    """Median and tail statistics of one group, as a flat dict."""
    def stats(values):
        values = np.asarray(values, dtype=np.float64)
        return dict(median=float(np.median(values)), p90=float(np.percentile(values, 90)),
                    mean=float(values.mean()), max=float(values.max()))

    return dict(count=int(len(report["latent_norm"])),
                latent_norm=stats(report["latent_norm"]),
                latent_percentile=stats(report["latent_percentile"]),
                fraction_above_99th=float((report["latent_percentile"] > 0.99).mean()),
                reconstruction_error=stats(report["reconstruction_error"]),
                noise_round_trip=stats(report["noise_round_trip"]),
                action_norm=stats(report["action_norm"]))
