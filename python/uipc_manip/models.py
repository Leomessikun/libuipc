"""Pure PyTorch PointNet++ encoder with the Wang RSS 2023 actor and critic heads.

The Newton dressing teacher builds its PointNet++ on ``torch_geometric``. That
dependency is not available in the Genesis environment, so the set-abstraction
blocks are implemented here on dense ``[B, N, C]`` tensors with a validity mask:
ball queries are a masked top-k over pairwise distances, feature aggregation is
a masked max, and padded points never contribute. With the reference ratios of
``1.0`` every valid point is a centroid, exactly as in the reference; ratios
below one use farthest point sampling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .obs import EXTRA_DIM, FEATURE_DIM, ObsSpec


@dataclass
class EncoderConfig:
    sa_radius: list[float] = field(default_factory=lambda: [0.05, 0.1])
    sa_ratio: list[float] = field(default_factory=lambda: [1.0, 1.0])
    sa_neighbors: list[int] = field(default_factory=lambda: [8, 16])
    """Neighbours kept per ball query. The dense query is O(neighbours), so this is the main cost knob."""
    sa_mlp: list[list[int]] = field(default_factory=lambda: [[64, 64, 128], [128, 128, 256], [256, 512, 1024]])
    linear_mlp: list[int] = field(default_factory=lambda: [128, 128])
    output_dim: int = 50

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EncoderConfig":
        return cls(**data)


def mlp(channels: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(1, len(channels)):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def gather_points(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather ``x[b, idx[b, s, k]]`` into ``[B, S, K, C]``."""
    b, s, k = idx.shape
    flat = idx.reshape(b, s * k, 1).expand(-1, -1, x.shape[-1])
    return torch.gather(x, 1, flat).reshape(b, s, k, x.shape[-1])


def farthest_point_sample(pos: torch.Tensor, valid: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Masked farthest point sampling; invalid points are never selected while valid ones remain."""
    b, n, _ = pos.shape
    device = pos.device
    distance = torch.full((b, n), float("inf"), device=device, dtype=pos.dtype)
    distance = distance.masked_fill(~valid, -1.0)
    farthest = valid.float().argmax(dim=1)
    batch_range = torch.arange(b, device=device)
    out = torch.zeros((b, n_samples), dtype=torch.long, device=device)
    for i in range(n_samples):
        out[:, i] = farthest
        centroid = pos[batch_range, farthest][:, None, :]
        d = ((pos - centroid) ** 2).sum(-1).masked_fill(~valid, -1.0)
        distance = torch.minimum(distance, d)
        farthest = distance.argmax(dim=1)
    return out


def ball_query(
    pos: torch.Tensor,
    valid: torch.Tensor,
    centers: torch.Tensor,
    center_valid: torch.Tensor,
    radius: float,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Up to ``k`` valid points within ``radius`` of every centre (nearest first)."""
    d = torch.cdist(centers, pos, compute_mode="donot_use_mm_for_euclid_dist")
    within = (d <= radius) & valid[:, None, :] & center_valid[:, :, None]
    score = torch.where(within, d, torch.full_like(d, float("inf")))
    k = min(int(k), pos.shape[1])
    values, idx = score.topk(k, dim=-1, largest=False)
    return idx, torch.isfinite(values)


class SetAbstraction(nn.Module):
    """PointNet++ set abstraction: ball query, shared MLP on relative coordinates, masked max."""

    def __init__(self, ratio: float, radius: float, k: int, in_channels: int, mlp_channels: list[int]) -> None:
        super().__init__()
        self.ratio = float(ratio)
        self.radius = float(radius)
        self.k = int(k)
        self.mlp = mlp([in_channels + 3, *mlp_channels])
        self.out_channels = int(mlp_channels[-1])

    def forward(self, x: torch.Tensor | None, pos: torch.Tensor, valid: torch.Tensor):
        if self.ratio < 1.0:
            n_samples = max(1, int(round(self.ratio * pos.shape[1])))
            idx = farthest_point_sample(pos, valid, n_samples)
            centers = torch.gather(pos, 1, idx[..., None].expand(-1, -1, 3))
            center_valid = torch.gather(valid, 1, idx)
        else:
            centers, center_valid = pos, valid
        nbr_idx, nbr_valid = ball_query(pos, valid, centers, center_valid, self.radius, self.k)
        rel = gather_points(pos, nbr_idx) - centers[:, :, None, :]
        feats = rel if x is None else torch.cat([gather_points(x, nbr_idx), rel], dim=-1)
        h = self.mlp(feats)
        h = h.masked_fill(~nbr_valid[..., None], float("-inf")).max(dim=2).values
        h = torch.where(center_valid[..., None] & torch.isfinite(h), h, torch.zeros_like(h))
        return h, centers, center_valid


class GlobalAbstraction(nn.Module):
    def __init__(self, in_channels: int, mlp_channels: list[int]) -> None:
        super().__init__()
        self.mlp = mlp([in_channels + 3, *mlp_channels])
        self.out_channels = int(mlp_channels[-1])

    def forward(self, x: torch.Tensor, pos: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        h = self.mlp(torch.cat([x, pos], dim=-1))
        h = h.masked_fill(~valid[..., None], float("-inf")).max(dim=1).values
        return torch.where(torch.isfinite(h), h, torch.zeros_like(h))


class PointNet2Encoder(nn.Module):
    """Two set-abstraction levels, one global level, then a small MLP (Wang ``RegressionNet``)."""

    def __init__(self, feature_dim: int, cfg: EncoderConfig) -> None:
        super().__init__()
        if len(cfg.sa_mlp) != 3 or len(cfg.sa_radius) < 2 or len(cfg.sa_ratio) < 2 or len(cfg.sa_neighbors) < 2:
            raise ValueError("EncoderConfig needs two local levels and one global level")
        self.sa1 = SetAbstraction(cfg.sa_ratio[0], cfg.sa_radius[0], cfg.sa_neighbors[0], feature_dim, cfg.sa_mlp[0])
        self.sa2 = SetAbstraction(
            cfg.sa_ratio[1], cfg.sa_radius[1], cfg.sa_neighbors[1], self.sa1.out_channels, cfg.sa_mlp[1]
        )
        self.global_sa = GlobalAbstraction(self.sa2.out_channels, cfg.sa_mlp[2])
        self.linear = mlp([self.global_sa.out_channels, *cfg.linear_mlp])
        self.out = nn.Linear(cfg.linear_mlp[-1] if cfg.linear_mlp else self.global_sa.out_channels, cfg.output_dim)
        self.feature_dim = int(cfg.output_dim)

    def forward(self, pos: torch.Tensor, feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x, p, v = self.sa1(feat, pos, valid)
        x, p, v = self.sa2(x, p, v)
        z = self.global_sa(x, p, v)
        return self.out(self.linear(z))


def _weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


def gaussian_logprob(noise: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu: torch.Tensor, pi: torch.Tensor | None, log_pi: torch.Tensor | None):
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi = log_pi - torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


class Actor(nn.Module):
    """Squashed Gaussian policy on ``[encoder(points), extra]``."""

    def __init__(
        self,
        spec: ObsSpec,
        action_dim: int,
        hidden_dim: int,
        encoder_cfg: EncoderConfig,
        use_extra: bool = True,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = PointNet2Encoder(FEATURE_DIM, encoder_cfg)
        self.use_extra = bool(use_extra)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        in_dim = self.encoder.feature_dim + (EXTRA_DIM if self.use_extra else 0)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim),
        )
        self.apply(_weight_init)

    def forward(self, obs, compute_pi: bool = True, compute_log_pi: bool = True, detach_encoder: bool = False):
        pos, feat, valid, extra = obs
        z = self.encoder(pos, feat, valid)
        if detach_encoder:
            z = z.detach()
        if self.use_extra:
            z = torch.cat([z, extra], dim=-1)
        mu, log_std = self.trunk(z).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi, noise = None, None
        log_pi = gaussian_logprob(noise, log_std) if (compute_log_pi and noise is not None) else None
        mu, pi, log_pi = squash(mu, pi, log_pi)
        return mu, pi, log_pi, log_std


class QHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.trunk(z)


class Critic(nn.Module):
    """Twin Q critic in the reference form ``Q(encode(s), a)``: the action joins after encoding."""

    def __init__(
        self, spec: ObsSpec, action_dim: int, hidden_dim: int, encoder_cfg: EncoderConfig, use_extra: bool = True
    ) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = PointNet2Encoder(FEATURE_DIM, encoder_cfg)
        self.use_extra = bool(use_extra)
        in_dim = self.encoder.feature_dim + int(action_dim) + (EXTRA_DIM if self.use_extra else 0)
        self.Q1 = QHead(in_dim, hidden_dim)
        self.Q2 = QHead(in_dim, hidden_dim)
        self.apply(_weight_init)

    def forward(self, obs, action: torch.Tensor, detach_encoder: bool = False):
        pos, feat, valid, extra = obs
        z = self.encoder(pos, feat, valid)
        if detach_encoder:
            z = z.detach()
        parts = [z, action]
        if self.use_extra:
            parts.append(extra)
        z = torch.cat(parts, dim=-1)
        return self.Q1(z), self.Q2(z)
