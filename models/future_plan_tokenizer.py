"""Motion-grounded tokenizer for future gesture plans.

The plan at time ``t`` is learned from a window beginning at
``t + future_plan_offset``. With the default four-frame RVQ rate, horizon two
and offset one describe the following eight motion frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _canonical_component_latent(value: torch.Tensor) -> torch.Tensor:
    """Convert cached residual-VQ latents to ``[B, T_token, D]``."""
    if value.ndim == 5:
        return value.sum(dim=(1, 2))
    if value.ndim == 4:
        return value.sum(dim=1)
    if value.ndim == 3:
        return value
    raise ValueError(f"unsupported cached RVQ latent shape: {tuple(value.shape)}")


def motion_plan_input_from_batch(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate coordinated upper-body, hand and lower-body latents."""
    upper = _canonical_component_latent(batch["zq_upper"])
    hands = _canonical_component_latent(batch["zq_hands"])
    lower = _canonical_component_latent(batch["zq_lower"])
    if upper.shape[:2] != hands.shape[:2] or upper.shape[:2] != lower.shape[:2]:
        raise ValueError("upper, hands, and lower RVQ token times must match")
    return torch.cat((upper, hands, lower), dim=-1)


def build_future_motion_windows(
    motion_features: torch.Tensor,
    horizon: int,
    offset: int,
    allow_current_plan: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened future windows and a full-window validity mask.

    Args:
        motion_features: Coordinated upper/hands/lower latent ``[B, T, D]``.
        horizon: Number of future RVQ tokens represented by one plan token.
        offset: First target offset. ``1`` deliberately excludes the current
            motion token and makes the representation predictive.

    Returns:
        Future windows ``[B, T, horizon * D]`` and ``[B, T]`` boolean mask.
    """
    if motion_features.ndim != 3:
        raise ValueError("motion_features must have shape [B, T, D]")
    if horizon <= 0:
        raise ValueError("future_plan_horizon must be positive")
    if offset < 0:
        raise ValueError("future_plan_offset must be non-negative")
    if offset == 0 and not allow_current_plan:
        raise ValueError(
            "future_plan_offset must be positive so that plans describe "
            "strictly future motion"
        )

    batch_size, tokens, channels = motion_features.shape
    padding = offset + horizon - 1
    padded = F.pad(motion_features, (0, 0, 0, padding))
    windows = torch.stack(
        [
            padded[:, offset + step : offset + step + tokens]
            for step in range(horizon)
        ],
        dim=2,
    )
    target_positions = (
        torch.arange(tokens, device=motion_features.device).unsqueeze(1)
        + offset
        + torch.arange(horizon, device=motion_features.device).unsqueeze(0)
    )
    valid_tokens = (target_positions < tokens).all(dim=1)
    valid_tokens = valid_tokens.unsqueeze(0).expand(batch_size, -1)
    return windows.reshape(batch_size, tokens, horizon * channels), valid_tokens


class ValidPlanVectorQuantizer(nn.Module):
    """Straight-through VQ whose loss and usage ignore padded future windows."""

    def __init__(self, num_codes: int, code_dim: int, commitment_weight: float = 0.25):
        super().__init__()
        self.num_codes = int(num_codes)
        self.code_dim = int(code_dim)
        self.commitment_weight = float(commitment_weight)
        self.codebook = nn.Parameter(torch.empty(self.num_codes, self.code_dim))
        nn.init.uniform_(self.codebook, -1.0 / self.num_codes, 1.0 / self.num_codes)

    def forward(
        self,
        latent: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if latent.ndim != 3 or latent.shape[-1] != self.code_dim:
            raise ValueError(
                f"expected [B, T, {self.code_dim}] latent, got {tuple(latent.shape)}"
            )
        if valid_mask.shape != latent.shape[:2]:
            raise ValueError("valid_mask must match the latent [B, T] dimensions")

        flat = latent.reshape(-1, self.code_dim)
        valid_flat = valid_mask.reshape(-1).bool()
        if not valid_flat.any():
            raise ValueError("future-plan window contains no fully valid target token")
        distance = (
            flat.square().sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.codebook.t()
            + self.codebook.square().sum(dim=1)
        )
        indices = distance.argmin(dim=1)
        quantized = F.embedding(indices, self.codebook).reshape_as(latent)
        valid_latent = latent.reshape(-1, self.code_dim)[valid_flat]
        valid_quantized = quantized.reshape(-1, self.code_dim)[valid_flat]
        commitment = F.mse_loss(valid_latent, valid_quantized.detach())
        codebook_loss = F.mse_loss(valid_quantized, valid_latent.detach())
        quantized_st = latent + (quantized - latent).detach()

        usage = F.one_hot(
            indices[valid_flat], num_classes=self.num_codes
        ).float().mean(dim=0)
        entropy = -(usage * usage.clamp_min(1e-8).log()).sum()
        return {
            "quantized": quantized_st,
            "indices": indices.reshape(latent.shape[:2]),
            "commitment_loss": self.commitment_weight * commitment + codebook_loss,
            "perplexity": entropy.exp(),
            "usage": usage,
            "valid_mask": valid_mask,
        }


class FutureWindowEncoder(nn.Module):
    """Encode each future window independently, without temporal leakage.

    A bidirectional Transformer over the plan-token timeline would let the
    code at ``t`` attend to an earlier overlapping window that contains the
    current motion.  Residual MLP blocks operate only on the flattened future
    window at ``t`` and preserve the intended predictive boundary.
    """

    def __init__(
        self,
        plan_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("plan_tokenizer_layers must be positive")
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(plan_dim),
                    nn.Linear(plan_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, plan_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(plan_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = value + block(value)
        return self.output_norm(value)


class FuturePlanTokenizer(nn.Module):
    """Learn discrete codes that reconstruct a following motion window."""

    def __init__(self, args=None):
        super().__init__()
        self.future_plan_horizon = int(getattr(args, "future_plan_horizon", 2))
        self.future_plan_offset = int(getattr(args, "future_plan_offset", 1))
        if self.future_plan_horizon <= 0:
            raise ValueError("future_plan_horizon must be positive")
        if self.future_plan_offset < 0:
            raise ValueError("future_plan_offset must be non-negative")
        if self.future_plan_offset == 0:
            raise ValueError("future_plan_offset must be positive")

        motion_dim = int(getattr(args, "motion_f", 256)) * 3
        input_dim = motion_dim * self.future_plan_horizon
        self.plan_dim = int(getattr(args, "intent_dim", 128))
        self.num_codes = int(getattr(args, "num_intent_tokens", 64))
        hidden_dim = int(getattr(args, "plan_tokenizer_hidden", 512))

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, self.plan_dim)
        self.encoder = FutureWindowEncoder(
            plan_dim=self.plan_dim,
            hidden_dim=hidden_dim,
            num_layers=int(getattr(args, "plan_tokenizer_layers", 2)),
            dropout=float(getattr(args, "intent_dropout", 0.1)),
        )
        self.quantizer = ValidPlanVectorQuantizer(
            self.num_codes,
            self.plan_dim,
            commitment_weight=float(getattr(args, "plan_vq_commitment", 0.25)),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(self.plan_dim),
            nn.Linear(self.plan_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def _prepare(self, batch_or_features) -> tuple[torch.Tensor, torch.Tensor]:
        motion_features = (
            motion_plan_input_from_batch(batch_or_features)
            if isinstance(batch_or_features, dict)
            else batch_or_features
        )
        return build_future_motion_windows(
            motion_features,
            horizon=self.future_plan_horizon,
            offset=self.future_plan_offset,
            allow_current_plan=False,
        )

    def encode(self, future_windows: torch.Tensor, valid_mask: torch.Tensor):
        encoded = self.encoder(
            self.input_projection(self.input_norm(future_windows))
        )
        return self.quantizer(encoded, valid_mask)

    @torch.no_grad()
    def encode_batch_with_mask(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return future-plan targets and positions with a complete horizon."""
        was_training = self.training
        self.eval()
        future_windows, valid_mask = self._prepare(batch)
        encoded = self.encode(future_windows, valid_mask)
        self.train(was_training)
        return encoded["indices"], valid_mask

    @torch.no_grad()
    def encode_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compatibility helper returning only future-plan code indices."""
        return self.encode_batch_with_mask(batch)[0]

    def forward(self, batch_or_features) -> dict[str, torch.Tensor]:
        future_windows, valid_mask = self._prepare(batch_or_features)
        quantized = self.encode(future_windows, valid_mask)
        reconstruction = self.decoder(quantized["quantized"])
        token_error = (reconstruction - future_windows).square().mean(dim=-1)
        reconstruction_loss = (
            token_error * valid_mask.to(token_error.dtype)
        ).sum() / valid_mask.sum().clamp_min(1)
        return {
            **quantized,
            "future_windows": future_windows,
            "reconstruction": reconstruction,
            "reconstruction_loss": reconstruction_loss,
            "total_loss": reconstruction_loss + quantized["commitment_loss"],
        }
