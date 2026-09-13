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

import contextlib
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .obs import EXTRA_DIM, FEATURE_DIM, FLAG_TOOL, ObsSpec


@dataclass
class EncoderConfig:
    kind: str = "pointnet2"
    """``pointnet2`` reproduces the reference; ``transformer`` is the attention alternative."""
    sa_radius: list[float] = field(default_factory=lambda: [0.05, 0.1])
    sa_ratio: list[float] = field(default_factory=lambda: [1.0, 1.0])
    sa_neighbors: list[int] = field(default_factory=lambda: [8, 16])
    """Neighbours kept per ball query. The dense query is O(neighbours), so this is the main cost knob."""
    sa_mlp: list[list[int]] = field(default_factory=lambda: [[64, 64, 128], [128, 128, 256], [256, 512, 1024]])
    fp_mlp: list[list[int]] = field(default_factory=lambda: [[256, 256], [256, 128], [128, 128, 128]])
    """Feature-propagation widths of the segmentation encoder (Wang ``fp_mlp_list``)."""
    linear_mlp: list[int] = field(default_factory=lambda: [128, 128])
    output_dim: int = 50
    transformer_dim: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 3

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


_NEIGHBOURHOOD_CACHE: dict | None = None


@contextlib.contextmanager
def reuse_neighbourhoods():
    """Compute each ball query once per set of position tensors inside the block.

    One SAC update runs the point encoders five to seven times on the same two
    observation batches, and a neighbourhood depends only on the positions, the
    validity masks, the radius and the count. Keys include the tensors' version
    counters, so an in-place change is never served from the cache.
    """
    global _NEIGHBOURHOOD_CACHE
    previous, _NEIGHBOURHOOD_CACHE = _NEIGHBOURHOOD_CACHE, {}
    try:
        yield
    finally:
        _NEIGHBOURHOOD_CACHE = previous


def ball_query(
    pos: torch.Tensor,
    valid: torch.Tensor,
    centers: torch.Tensor,
    center_valid: torch.Tensor,
    radius: float,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Up to ``k`` valid points within ``radius`` of every centre (nearest first).

    Distances are compared squared, from explicit coordinate differences. The
    previous ``torch.cdist`` without its matrix-product path took 72% of a SAC
    update's GPU time on these three-dimensional distances; the neighbour sets
    and validity are the same, see the dressing-correctness record.
    """
    cache = _NEIGHBOURHOOD_CACHE
    if cache is not None:
        key = tuple((t.data_ptr(), t._version, tuple(t.shape)) for t in (pos, valid, centers, center_valid)) + (float(radius), int(k))
        hit = cache.get(key)
        if hit is not None:
            return hit
    d2 = (
        (centers[:, :, None, 0] - pos[:, None, :, 0]) ** 2
        + (centers[:, :, None, 1] - pos[:, None, :, 1]) ** 2
        + (centers[:, :, None, 2] - pos[:, None, :, 2]) ** 2
    )
    within = (d2 <= float(radius) * float(radius)) & valid[:, None, :] & center_valid[:, :, None]
    score = torch.where(within, d2, torch.full_like(d2, float("inf")))
    k = min(int(k), pos.shape[1])
    values, idx = score.topk(k, dim=-1, largest=False)
    result = (idx, torch.isfinite(values))
    if cache is not None:
        cache[key] = result
    return result


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


def squashed_action_log_prob(mu: torch.Tensor, log_std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Log-density of an executed ``tanh``-squashed action under the pre-squash Gaussian.

    The inverse of :func:`squash`: recover the pre-tanh sample by ``atanh``,
    score it under ``N(mu, exp(log_std))``, and subtract the squashing
    log-determinant. Actions are clamped just inside ``(-1, 1)`` so saturated
    targets stay finite; their density is then meaningless, which is why
    behaviour cloning from a bang-bang scripted policy should use ``mse``.
    """
    squashed = action.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
    pre_tanh = 0.5 * (torch.log1p(squashed) - torch.log1p(-squashed))
    noise = (pre_tanh - mu) / log_std.exp().clamp_min(1.0e-8)
    log_pi = gaussian_logprob(noise, log_std)
    return log_pi - torch.log(F.relu(1.0 - squashed.pow(2)) + 1e-6).sum(-1, keepdim=True)


def _bounded_log_std(log_std: torch.Tensor, low: float, high: float) -> torch.Tensor:
    log_std = torch.tanh(log_std)
    return low + 0.5 * (high - low) * (log_std + 1)


def _sample_head(mu: torch.Tensor, log_std: torch.Tensor, compute_pi: bool, compute_log_pi: bool):
    if compute_pi:
        std = log_std.exp()
        noise = torch.randn_like(mu)
        pi = mu + noise * std
    else:
        pi, noise = None, None
    log_pi = gaussian_logprob(noise, log_std) if (compute_log_pi and noise is not None) else None
    mu, pi, log_pi = squash(mu, pi, log_pi)
    return mu, pi, log_pi, log_std


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
        trunk_style: str = "plain",
        trunk_blocks: int = 2,
        history_length: int = 1,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = make_global_encoder(FEATURE_DIM, encoder_cfg)
        self.use_extra = bool(use_extra)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        frame_dim = self.encoder.feature_dim + (EXTRA_DIM if self.use_extra else 0)
        self.history, in_dim = make_history(frame_dim, action_dim, history_length)
        self.trunk = trunk_layers(in_dim, hidden_dim, 2 * action_dim, trunk_style, trunk_blocks)
        self.apply(_weight_init)

    def _frame_latent(self, obs, detach_encoder: bool = False) -> torch.Tensor:
        pos, feat, valid, extra = obs
        z = self.encoder(pos, feat, valid)
        if detach_encoder:
            z = z.detach()
        return torch.cat([z, extra], dim=-1) if self.use_extra else z

    def head(self, obs, detach_encoder: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-squash mean and bounded log standard deviation."""
        z = history_input(self.history, obs, lambda frames: self._frame_latent(frames, detach_encoder))
        mu, log_std = self.trunk(z).chunk(2, dim=-1)
        return mu, _bounded_log_std(log_std, self.log_std_min, self.log_std_max)

    def forward(self, obs, compute_pi: bool = True, compute_log_pi: bool = True, detach_encoder: bool = False):
        mu, log_std = self.head(obs, detach_encoder)
        return _sample_head(mu, log_std, compute_pi, compute_log_pi)

    def action_log_prob(self, obs, action: torch.Tensor, detach_encoder: bool = False) -> torch.Tensor:
        mu, log_std = self.head(obs, detach_encoder)
        return squashed_action_log_prob(mu, log_std, action)


class ResidualBlock(nn.Module):
    """Pre-normalised residual block: ``x + W2 ReLU(W1 LayerNorm(x))``.

    The shape BroNet and SimBa converge on for value networks. Without normalisation a plain
    multi-layer perceptron critic is reported to get *worse* as it is made larger, while a
    normalised residual one improves, which is why width is not worth buying on its own.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))


def trunk_layers(in_dim: int, hidden_dim: int, out_dim: int, style: str, blocks: int = 2) -> nn.Sequential:
    """The body of a head, in one of two shapes.

    ``plain`` is this port's original ``Linear-ReLU-Linear-ReLU-Linear``. ``residual`` embeds into
    ``hidden_dim``, applies ``blocks`` pre-normalised residual blocks, normalises once more and
    projects out — the arrangement reported to be what lets a value network benefit from scale.
    """
    if style == "plain":
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    if style != "residual":
        raise ValueError(f"Unknown trunk style {style!r}")
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        *[ResidualBlock(hidden_dim) for _ in range(int(blocks))],
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, out_dim),
    )


class FrameHistory(nn.Module):
    """Ordered finite history: spatial encoding first, temporal concatenation after.

    Point order is not time, so each cloud is encoded on its own and only the
    resulting frame vectors are concatenated in decision order, together with the
    commands recorded between them. Padded frames contribute zeros and carry a
    validity flag, so the empty history at an episode opening stays distinguishable
    from a frame that happens to encode to zero. The module holds no parameters:
    it defines the layout the trunk reads, and a target copy has nothing to track.

    ``action_dim = 0`` means the commands are already inside the frame encodings,
    as they are in the dense critic, where each frame is encoded with its own
    command as a per-point feature.
    """

    def __init__(self, frame_dim: int, action_dim: int, length: int) -> None:
        super().__init__()
        self.length, self.frame_dim, self.action_dim = int(length), int(frame_dim), int(action_dim)
        if self.length < 2:
            raise ValueError("A frame history needs at least two frames; length 1 is the single-frame policy")
        self.out_dim = self.length * (self.frame_dim + 1) + (self.length - 1) * self.action_dim

    def forward(self, latent: torch.Tensor, valid: torch.Tensor, commands: torch.Tensor | None = None) -> torch.Tensor:
        """``latent [B,L,frame_dim]``, ``valid [B,L]``, ``commands [B,L-1,action_dim]`` -> ``[B,out_dim]``."""
        if latent.shape[1] != self.length:
            raise ValueError(f"This history holds {self.length} frames, got {latent.shape[1]}")
        mask = valid.to(latent.dtype).unsqueeze(-1)
        frames = torch.cat([latent * mask, mask], dim=-1).flatten(1)
        if not self.action_dim:
            return frames
        # A command is valid exactly where the observation it followed is.
        return torch.cat([frames, (commands * mask[:, :-1]).flatten(1)], dim=-1)


def make_history(frame_dim: int, action_dim: int, length: int) -> tuple[FrameHistory | None, int]:
    """``(history, trunk input width)``. ``length = 1`` keeps the single-frame network exactly."""
    if int(length) == 1:
        return None, int(frame_dim)
    memory = FrameHistory(frame_dim, action_dim, length)
    return memory, memory.out_dim


def history_input(history: FrameHistory | None, obs, frame_latent) -> torch.Tensor:
    """Trunk input for one frame, or for a window encoded frame by frame.

    Without history ``obs`` is the usual ``(pos, feat, valid, extra)``. With it,
    ``obs`` is ``(frames, valid, commands)``: the same four tensors flattened to
    ``B*L`` rows, the ``[B,L]`` mask of recorded frames, and the ``[B,L-1,A]``
    commands between them.
    """
    if history is None:
        return frame_latent(obs)
    frames, valid, commands = obs
    latent = frame_latent(frames)
    return history(latent.reshape(*valid.shape, -1), valid, commands)


class QHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, trunk_style: str = "plain", blocks: int = 2) -> None:
        super().__init__()
        self.trunk = trunk_layers(in_dim, hidden_dim, 1, trunk_style, blocks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.trunk(z)


def _broadcast_action(feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Append the action to every point's features, as the reference's Q function does.

    Wang RSS 2023: "we concatenate a* as an additional feature to every point p_i in the input point
    cloud, so the feature of each point includes its 3D position, the one-hot vector of the class
    type, and the action a*. We then use a classification-type neural network architecture to output
    a scalar Q value." Their own Table I measures the alternative, encoding the cloud first and
    concatenating the action to the latent vector, at about 0.11 lower upper-arm dressed ratio.
    """
    return torch.cat([feat, action.unsqueeze(1).expand(-1, feat.shape[1], -1)], dim=-1)


class Critic(nn.Module):
    """Twin Q critic. ``action_mode`` selects where the action enters the network.

    ``dense`` is the reference's: the action becomes a per-point feature and the encoder sees it, so
    the encoding itself is a function of the action. ``latent`` encodes the cloud alone and
    concatenates the action afterwards, which is the baseline the reference reports as much worse
    (``agent_docs/performance/2026-09-12-critic-architecture-defect.md``); it is kept so checkpoints
    written before the fix still load and so the two can be compared directly.
    """

    def __init__(
        self, spec: ObsSpec, action_dim: int, hidden_dim: int, encoder_cfg: EncoderConfig, use_extra: bool = True,
        action_mode: str = "dense", trunk_style: str = "plain", trunk_blocks: int = 2, history_length: int = 1,
    ) -> None:
        super().__init__()
        if action_mode not in ("dense", "latent"):
            raise ValueError(f"Unknown critic action_mode {action_mode!r}")
        if int(history_length) != 1 and action_mode != "dense":
            # The latent critic is kept only for old checkpoints and its ablation; a history on top
            # of the rejected baseline would measure two things at once.
            raise ValueError("A history-aware critic requires action_mode='dense'")
        self.spec = spec
        self.action_mode = str(action_mode)
        self.action_dim = int(action_dim)
        extra_point_features = self.action_dim if self.action_mode == "dense" else 0
        self.encoder = make_global_encoder(FEATURE_DIM + extra_point_features, encoder_cfg)
        self.use_extra = bool(use_extra)
        # Under ``dense`` the action is already inside the encoding, so the head does not take it again.
        frame_dim = self.encoder.feature_dim + (0 if self.action_mode == "dense" else self.action_dim)
        frame_dim += EXTRA_DIM if self.use_extra else 0
        # Past commands enter each earlier frame the way the candidate enters the current one, so the
        # history layout carries no separate command block.
        self.history, in_dim = make_history(frame_dim, 0, history_length)
        self.Q1 = QHead(in_dim, hidden_dim, trunk_style, trunk_blocks)
        self.Q2 = QHead(in_dim, hidden_dim, trunk_style, trunk_blocks)
        self.apply(_weight_init)

    def encode(self, obs, action: torch.Tensor) -> torch.Tensor:
        """Encode the observation as this critic's Q heads see it.

        Under ``dense`` the encoder's input carries the action on every point, so its width is
        ``FEATURE_DIM + action_dim`` and the encoding is a function of the action; callers must go
        through here rather than call ``self.encoder`` with the raw features.
        """
        pos, feat, valid, _ = obs
        if self.action_mode == "dense":
            feat = _broadcast_action(feat, action)
        return self.encoder(pos, feat, valid)

    def _frame_latent(self, obs, action: torch.Tensor, detach_encoder: bool = False) -> torch.Tensor:
        z = self.encode(obs, action)
        if detach_encoder:
            z = z.detach()
        parts = [z] if self.action_mode == "dense" else [z, action]
        if self.use_extra:
            parts.append(obs[3])
        return torch.cat(parts, dim=-1)

    def forward(self, obs, action: torch.Tensor, detach_encoder: bool = False):
        """``Q(history, action)``: the candidate action is a per-point feature of the current frame only.

        With history, every earlier frame is encoded with the command that was actually recorded
        after it. Scoring a candidate therefore never rewrites the observed past, and ``dQ/da``
        flows through the current frame's encoding exactly as in the single-frame critic.
        """
        if self.history is None:
            z = self._frame_latent(obs, action, detach_encoder)
        else:
            frames, valid, commands = obs
            per_frame = torch.cat([commands, action.unsqueeze(1)], dim=1).reshape(-1, self.action_dim)
            latent = self._frame_latent(frames, per_frame, detach_encoder)
            z = self.history(latent.reshape(*valid.shape, -1), valid)
        return self.Q1(z), self.Q2(z)



class PrivilegedCritic(nn.Module):
    """Twin Q critic on the simulator's low-dimensional state, ``Q(s_priv, a)``.

    The asymmetric actor-critic of Pinto et al. (2018): the actor keeps the
    point cloud, and only the critic, which training discards, reads the
    privileged state. With no point encoder, the critic's forward and backward
    passes drop out of the update's point work, leaving the actor's.
    """

    encoder = None

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int,
                 trunk_style: str = "plain", trunk_blocks: int = 2) -> None:
        super().__init__()
        in_dim = int(state_dim) + int(action_dim)
        self.Q1 = QHead(in_dim, hidden_dim, trunk_style, trunk_blocks)
        self.Q2 = QHead(in_dim, hidden_dim, trunk_style, trunk_blocks)
        self.apply(_weight_init)

    def forward(self, state: torch.Tensor, action: torch.Tensor, detach_encoder: bool = False):
        z = torch.cat([state, action], dim=-1)
        return self.Q1(z), self.Q2(z)

# ---------------------------------------------------------------------------
# Segmentation PointNet++ (per-point features) for the Wang flow actor
# ---------------------------------------------------------------------------


class PointNet2Segmentation(nn.Module):
    """Wang ``SegmentationNet``: PointNet++ with feature propagation back to every point.

    The reference propagates coarse features to fine levels with inverse-distance
    k-nearest-neighbour interpolation. With the reference ratios of ``1.0`` every
    level holds the same points, so each point's nearest neighbour is itself at
    distance zero and the ``1 / d^2`` weight of that neighbour dominates the other
    two by many orders of magnitude. The interpolation is therefore the identity
    map and is implemented as direct concatenation, which is the same shortcut
    the Newton port takes with ``direct_index``. Ratios below one are rejected
    here rather than silently approximated.
    """

    def __init__(self, feature_dim: int, cfg: EncoderConfig) -> None:
        super().__init__()
        if any(float(r) < 1.0 for r in cfg.sa_ratio[:2]):
            raise ValueError("PointNet2Segmentation implements the identity propagation of ratio 1.0 only")
        self.sa1 = SetAbstraction(1.0, cfg.sa_radius[0], cfg.sa_neighbors[0], feature_dim, cfg.sa_mlp[0])
        self.sa2 = SetAbstraction(1.0, cfg.sa_radius[1], cfg.sa_neighbors[1], self.sa1.out_channels, cfg.sa_mlp[1])
        self.global_sa = GlobalAbstraction(self.sa2.out_channels, cfg.sa_mlp[2])
        fp = cfg.fp_mlp
        self.fp0 = mlp([self.global_sa.out_channels + self.sa2.out_channels, *fp[0]])
        self.fp1 = mlp([fp[0][-1] + self.sa1.out_channels, *fp[1]])
        self.fp2 = mlp([fp[1][-1] + feature_dim, *fp[2]])
        self.linear = mlp([fp[2][-1], *cfg.linear_mlp])
        self.out = nn.Linear(cfg.linear_mlp[-1] if cfg.linear_mlp else fp[2][-1], cfg.output_dim)
        self.feature_dim = int(cfg.output_dim)

    def forward(self, pos: torch.Tensor, feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Per-point features ``[B, N, output_dim]``; padded rows are zero."""
        x1, _, _ = self.sa1(feat, pos, valid)
        x2, _, _ = self.sa2(x1, pos, valid)
        g = self.global_sa(x2, pos, valid)
        h = self.fp0(torch.cat([g[:, None, :].expand(-1, pos.shape[1], -1), x2], dim=-1))
        h = self.fp1(torch.cat([h, x1], dim=-1))
        h = self.fp2(torch.cat([h, feat], dim=-1))
        h = self.out(self.linear(h))
        return h * valid[..., None]


# ---------------------------------------------------------------------------
# Set transformer encoder (a modern alternative to PointNet++)
# ---------------------------------------------------------------------------


class SetTransformerEncoder(nn.Module):
    """Self-attention over points with a learned global token.

    Every point becomes a token from its tool-relative position and segmentation
    flags; padded rows are masked out of attention. The global token's output is
    the global feature and each point's output is its per-point feature, so one
    forward serves both the critic and the tool-point actor readout. Attention
    over a few hundred points is dense and cheap, and unlike a ball query it has
    no radius to tune against the scene scale.
    """

    def __init__(self, feature_dim: int, cfg: EncoderConfig) -> None:
        super().__init__()
        d = int(cfg.transformer_dim)
        self.embed = nn.Sequential(nn.Linear(3 + feature_dim, d), nn.ReLU(), nn.Linear(d, d))
        self.global_token = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(cfg.transformer_heads),
            dim_feedforward=4 * d,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=int(cfg.transformer_layers), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, cfg.output_dim)
        self.feature_dim = int(cfg.output_dim)

    def _run(self, pos: torch.Tensor, feat: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.embed(torch.cat([pos, feat], dim=-1))
        tokens = torch.cat([self.global_token.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        mask = torch.cat([torch.zeros_like(valid[:, :1]), ~valid], dim=1)
        h = self.norm(self.blocks(tokens, src_key_padding_mask=mask))
        return h[:, 0], h[:, 1:] * valid[..., None]

    def forward(self, pos: torch.Tensor, feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.out(self._run(pos, feat, valid)[0])

    def point_features(self, pos: torch.Tensor, feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.out(self._run(pos, feat, valid)[1]) * valid[..., None]


def make_global_encoder(feature_dim: int, cfg: EncoderConfig) -> nn.Module:
    if cfg.kind == "pointnet2":
        return PointNet2Encoder(feature_dim, cfg)
    if cfg.kind == "transformer":
        return SetTransformerEncoder(feature_dim, cfg)
    raise ValueError(f"Unknown encoder kind {cfg.kind!r}")


class _PointFeatureAdapter(nn.Module):
    """Expose ``point_features`` of a transformer through the segmentation interface."""

    def __init__(self, encoder: SetTransformerEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.feature_dim = encoder.feature_dim

    def forward(self, pos: torch.Tensor, feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.encoder.point_features(pos, feat, valid)


def make_point_encoder(feature_dim: int, cfg: EncoderConfig) -> nn.Module:
    if cfg.kind == "pointnet2":
        return PointNet2Segmentation(feature_dim, cfg)
    if cfg.kind == "transformer":
        return _PointFeatureAdapter(SetTransformerEncoder(feature_dim, cfg))
    raise ValueError(f"Unknown encoder kind {cfg.kind!r}")


class WangFlowActor(nn.Module):
    """Wang RSS 2023 policy head: read the per-point feature at the explicit tool point.

    This is the actor the Newton ``--fmvp-pretrain-defaults`` preset trains. The
    segmentation encoder gives every point a feature; the policy trunk sees only
    the tool point's row, which localises the policy at the gripper. ``use_extra``
    additionally concatenates the observation tail (absolute tool position and
    goal offset), which the reference does not have because its task carries no
    separate goal.
    """

    def __init__(
        self,
        spec: ObsSpec,
        action_dim: int,
        hidden_dim: int,
        encoder_cfg: EncoderConfig,
        use_extra: bool = True,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        trunk_style: str = "plain",
        trunk_blocks: int = 2,
        history_length: int = 1,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = make_point_encoder(FEATURE_DIM, encoder_cfg)
        self.use_extra = bool(use_extra)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        frame_dim = self.encoder.feature_dim + (EXTRA_DIM if self.use_extra else 0)
        self.history, in_dim = make_history(frame_dim, action_dim, history_length)
        self.trunk = trunk_layers(in_dim, hidden_dim, 2 * action_dim, trunk_style, trunk_blocks)
        self.apply(_weight_init)

    def _frame_latent(self, obs, detach_encoder: bool = False) -> torch.Tensor:
        pos, feat, valid, extra = obs
        point_features = self.encoder(pos, feat, valid)
        if detach_encoder:
            point_features = point_features.detach()
        tool_index = (feat[:, :, FLAG_TOOL] * valid).argmax(dim=1)
        z = point_features[torch.arange(pos.shape[0], device=pos.device), tool_index]
        return torch.cat([z, extra], dim=-1) if self.use_extra else z

    def head(self, obs, detach_encoder: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-squash mean and bounded log standard deviation read at the tool point."""
        z = history_input(self.history, obs, lambda frames: self._frame_latent(frames, detach_encoder))
        mu, log_std = self.trunk(z).chunk(2, dim=-1)
        return mu, _bounded_log_std(log_std, self.log_std_min, self.log_std_max)

    def forward(self, obs, compute_pi: bool = True, compute_log_pi: bool = True, detach_encoder: bool = False):
        mu, log_std = self.head(obs, detach_encoder)
        return _sample_head(mu, log_std, compute_pi, compute_log_pi)

    def action_log_prob(self, obs, action: torch.Tensor, detach_encoder: bool = False) -> torch.Tensor:
        mu, log_std = self.head(obs, detach_encoder)
        return squashed_action_log_prob(mu, log_std, action)


# ---------------------------------------------------------------------------
# Categorical (C51-style) critic for the FlashSAC path
# ---------------------------------------------------------------------------


class CategoricalQHead(nn.Module):
    """Distributional Q head over ``num_bins`` value atoms in ``[min_v, max_v]``.

    Bounding the atoms bounds the expected value structurally, which is what the
    Newton FlashSAC path relies on to stop scalar Q from chasing an unbounded
    steady state on dense negative rewards.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_bins: int, min_v: float, max_v: float,
                 trunk_style: str = "plain", blocks: int = 2) -> None:
        super().__init__()
        self.trunk = trunk_layers(in_dim, hidden_dim, int(num_bins), trunk_style, blocks)
        self.register_buffer("bin_values", torch.linspace(float(min_v), float(max_v), int(num_bins)))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = F.log_softmax(self.trunk(z), dim=-1)
        expected = (log_probs.exp() * self.bin_values).sum(dim=-1, keepdim=True)
        return expected, log_probs


class CategoricalCritic(nn.Module):
    """Twin distributional critic; ``action_mode`` is :class:`Critic`'s, with the same meaning."""

    def __init__(
        self,
        spec: ObsSpec,
        action_dim: int,
        hidden_dim: int,
        encoder_cfg: EncoderConfig,
        num_bins: int,
        min_v: float,
        max_v: float,
        use_extra: bool = True,
        action_mode: str = "dense",
        trunk_style: str = "plain",
        trunk_blocks: int = 2,
    ) -> None:
        super().__init__()
        if action_mode not in ("dense", "latent"):
            raise ValueError(f"Unknown critic action_mode {action_mode!r}")
        self.spec = spec
        self.action_mode = str(action_mode)
        self.action_dim = int(action_dim)
        extra_point_features = self.action_dim if self.action_mode == "dense" else 0
        self.encoder = make_global_encoder(FEATURE_DIM + extra_point_features, encoder_cfg)
        self.use_extra = bool(use_extra)
        self.num_bins, self.min_v, self.max_v = int(num_bins), float(min_v), float(max_v)
        in_dim = self.encoder.feature_dim + (0 if self.action_mode == "dense" else self.action_dim)
        in_dim += EXTRA_DIM if self.use_extra else 0
        self.Q1 = CategoricalQHead(in_dim, hidden_dim, num_bins, min_v, max_v, trunk_style, trunk_blocks)
        self.Q2 = CategoricalQHead(in_dim, hidden_dim, num_bins, min_v, max_v, trunk_style, trunk_blocks)
        self.apply(_weight_init)

    def encode(self, obs, action: torch.Tensor) -> torch.Tensor:
        """Encode as the Q heads see it; see :meth:`Critic.encode`."""
        pos, feat, valid, _ = obs
        if self.action_mode == "dense":
            feat = _broadcast_action(feat, action)
        return self.encoder(pos, feat, valid)

    def forward(self, obs, action: torch.Tensor, detach_encoder: bool = False):
        z = self.encode(obs, action)
        if detach_encoder:
            z = z.detach()
        parts = [z] if self.action_mode == "dense" else [z, action]
        if self.use_extra:
            parts.append(obs[3])
        z = torch.cat(parts, dim=-1)
        q1, log_p1 = self.Q1(z)
        q2, log_p2 = self.Q2(z)
        return q1, q2, log_p1, log_p2
