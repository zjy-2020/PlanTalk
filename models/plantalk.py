"""PlanTalk: future-grounded planning for holistic co-speech motion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .backbone import HolisticBodyGenerator
except ImportError:
    from models.backbone import HolisticBodyGenerator


class TemporalFiLM(nn.Module):
    """Apply a token-rate plan to a sequence at an arbitrary time resolution."""

    def __init__(self, plan_dim: int, feature_dim: int):
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.LayerNorm(plan_dim),
            nn.Linear(plan_dim, 2 * feature_dim),
        )
        nn.init.zeros_(self.to_scale_shift[-1].weight)
        nn.init.zeros_(self.to_scale_shift[-1].bias)

    def forward(self, features: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
        """Return FiLM-modulated ``features``.

        Args:
            features: ``[batch, time, feature_dim]`` sequence to condition.
            plan: ``[batch, plan_time, plan_dim]`` plan sequence.
        """
        if features.ndim != 3 or plan.ndim != 3:
            raise ValueError("TemporalFiLM expects [B, T, C] feature and plan tensors")
        if features.shape[0] != plan.shape[0]:
            raise ValueError("feature and plan batch sizes must match")

        if plan.shape[1] != features.shape[1]:
            plan = F.interpolate(
                plan.transpose(1, 2),
                size=features.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        scale, shift = self.to_scale_shift(plan).chunk(2, dim=-1)
        return features * (1.0 + torch.tanh(scale)) + shift


class SpeechPlanPredictor(nn.Module):
    """Predict discrete communicative plans from speech-semantic features.

    The planner operates at the four-frame RVQ token rate. Its logits are mapped
    to a frozen, motion-grounded codebook during both training and inference.
    """

    def __init__(
        self,
        semantic_dim: int = 256,
        plan_dim: int = 128,
        num_plan_tokens: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if plan_dim % num_heads != 0:
            raise ValueError("intent_dim must be divisible by intent_num_heads")

        self.semantic_to_plan = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, plan_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=plan_dim,
            nhead=num_heads,
            dim_feedforward=plan_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(plan_dim)
        self.token_classifier = nn.Linear(plan_dim, num_plan_tokens)

    def forward(self, semantic_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if semantic_tokens.ndim != 3:
            raise ValueError("semantic_tokens must have shape [B, T_plan, C]")
        plan_feature = self.output_norm(
            self.transformer(self.semantic_to_plan(semantic_tokens))
        )
        return plan_feature, self.token_classifier(plan_feature)


class PlanTalk(HolisticBodyGenerator):
    """Plan-conditioned holistic co-speech motion generator."""

    def __init__(self, args=None):
        super().__init__(args)
        self.intent_dim = int(getattr(args, "intent_dim", 128))
        self.num_intent_tokens = int(getattr(args, "num_intent_tokens", 64))
        planner_heads = int(getattr(args, "intent_num_heads", 4))
        planner_layers = int(getattr(args, "intent_num_layers", 2))
        planner_dropout = float(getattr(args, "intent_dropout", 0.1))

        self.intent_planner = SpeechPlanPredictor(
            semantic_dim=args.audio_f,
            plan_dim=self.intent_dim,
            num_plan_tokens=self.num_intent_tokens,
            num_layers=planner_layers,
            num_heads=planner_heads,
            dropout=planner_dropout,
        )

        # One global body modulation plus independent body-part modulations.
        self.frame_plan_film = TemporalFiLM(self.intent_dim, args.hidden_size)
        self.upper_plan_film = TemporalFiLM(self.intent_dim, args.hidden_size)
        self.hands_plan_film = TemporalFiLM(self.intent_dim, args.hidden_size)
        self.lower_plan_film = TemporalFiLM(self.intent_dim, args.hidden_size)
        self.upper_token_plan_film = TemporalFiLM(self.intent_dim, args.audio_f)
        self.hands_token_plan_film = TemporalFiLM(self.intent_dim, args.audio_f)
        self.lower_token_plan_film = TemporalFiLM(self.intent_dim, args.audio_f)
        self.plan_enabled = True
        self.plan_temperature = float(getattr(args, "plan_temperature", 1.0))
        self.plan_selection = str(
            getattr(args, "plan_selection", "st_gumbel")
        ).lower()
        self.plan_eval_selection = str(
            getattr(args, "plan_eval_selection", "argmax")
        ).lower()
        if self.plan_selection not in {"soft", "st_gumbel", "argmax"}:
            raise ValueError("plan_selection must be soft, st_gumbel, or argmax")
        if self.plan_eval_selection not in {"soft", "argmax", "sample"}:
            raise ValueError("plan_eval_selection must be soft, argmax, or sample")

        self.register_buffer(
            "motion_plan_codebook",
            torch.zeros(self.num_intent_tokens, self.intent_dim),
        )
        self._plan_codebook_loaded = False

        semantic_dim = int(getattr(args, "audio_f", 256))
        self.plan_gate_strength = float(getattr(args, "plan_gate_strength", 1.0))
        self.event_planner = nn.Sequential(
            nn.LayerNorm(semantic_dim + self.intent_dim),
            nn.Linear(semantic_dim + self.intent_dim, self.intent_dim),
            nn.GELU(),
            nn.Dropout(planner_dropout),
            nn.Linear(self.intent_dim, 1),
        )
        self.plan_gate_delta = nn.Sequential(
            nn.LayerNorm(self.intent_dim),
            nn.Linear(self.intent_dim, 2),
        )
        nn.init.zeros_(self.event_planner[-1].weight)
        nn.init.zeros_(self.event_planner[-1].bias)
        nn.init.zeros_(self.plan_gate_delta[-1].weight)
        nn.init.zeros_(self.plan_gate_delta[-1].bias)
        self._last_plan_diagnostics = {}
        self._last_gate_diagnostics = {}

    def set_plan_enabled(self, enabled: bool) -> None:
        """Enable or disable discrete planning without changing decoder shapes."""
        self.plan_enabled = bool(enabled)

    @torch.no_grad()
    def set_motion_plan_codebook(self, codebook: torch.Tensor) -> None:
        """Copy a pretrained future-motion codebook into the generator."""
        if codebook.shape != self.motion_plan_codebook.shape:
            raise ValueError(
                "motion-plan codebook shape mismatch: "
                f"expected {tuple(self.motion_plan_codebook.shape)}, "
                f"got {tuple(codebook.shape)}"
            )
        self.motion_plan_codebook.copy_(
            codebook.to(
                device=self.motion_plan_codebook.device,
                dtype=self.motion_plan_codebook.dtype,
            )
        )
        self._plan_codebook_loaded = True

    def _select_plan_weights(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return selected code weights and the underlying soft distribution."""
        temperature = max(self.plan_temperature, 1e-6)
        probabilities = torch.softmax(logits / temperature, dim=-1)
        if self.training:
            if self.plan_selection == "st_gumbel":
                weights = F.gumbel_softmax(
                    logits,
                    tau=temperature,
                    hard=True,
                    dim=-1,
                )
                return weights, probabilities
            if self.plan_selection == "argmax":
                hard = F.one_hot(
                    logits.argmax(dim=-1),
                    num_classes=self.num_intent_tokens,
                ).to(probabilities.dtype)
                return hard, probabilities
            return probabilities, probabilities

        if self.plan_eval_selection == "soft":
            return probabilities, probabilities
        if self.plan_eval_selection == "sample":
            indices = torch.multinomial(
                probabilities.reshape(-1, self.num_intent_tokens),
                num_samples=1,
            ).reshape(logits.shape[:-1])
        else:
            indices = probabilities.argmax(dim=-1)
        weights = F.one_hot(indices, num_classes=self.num_intent_tokens).to(logits.dtype)
        return weights, probabilities

    def _predict_plan(
        self,
        semantic_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict discrete plans and retrieve their motion-grounded features."""
        batch_size, tokens, _ = semantic_tokens.shape
        if not self.plan_enabled:
            feature = semantic_tokens.new_zeros(batch_size, tokens, self.intent_dim)
            logits = semantic_tokens.new_zeros(
                batch_size, tokens, self.num_intent_tokens
            )
            self._last_plan_diagnostics = {
                "plan_probabilities": logits,
                "plan_soft_probabilities": logits,
                "plan_token_ids": logits.new_zeros(
                    batch_size, tokens, dtype=torch.long
                ),
                "speech_plan_context": feature,
            }
            return feature, logits
        if not self._plan_codebook_loaded:
            raise RuntimeError(
                "The future-motion codebook has not been loaded. "
                "Call set_motion_plan_codebook() before training or inference."
            )

        speech_plan_context, logits = self.intent_planner(semantic_tokens)
        weights, probabilities = self._select_plan_weights(logits)
        feature = torch.matmul(weights, self.motion_plan_codebook)
        self._last_plan_diagnostics = {
            "plan_probabilities": weights,
            "plan_soft_probabilities": probabilities,
            "plan_token_ids": weights.argmax(dim=-1),
            "speech_plan_context": speech_plan_context,
        }
        return feature, logits

    def _predict_gate(
        self,
        semantic_tokens: torch.Tensor,
        plan_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse frame semantics with plan-conditioned event activation."""
        semantic_gate = self.gate(semantic_tokens)
        if not self.plan_enabled:
            event_logits = semantic_tokens.new_zeros(semantic_tokens.shape[:2])
            combined_gate = semantic_gate
        else:
            event_logits = self.event_planner(
                torch.cat((semantic_tokens, plan_feature), dim=-1)
            ).squeeze(-1)
            delta = torch.tanh(self.plan_gate_delta(plan_feature))
            event_margin = torch.stack(
                (-0.5 * torch.tanh(event_logits), 0.5 * torch.tanh(event_logits)),
                dim=-1,
            )
            combined_gate = semantic_gate + self.plan_gate_strength * (
                delta + event_margin
            )
        self._last_gate_diagnostics = {
            "base_gate": semantic_gate,
            "event_logits": event_logits,
            "event_probabilities": torch.sigmoid(event_logits),
            "plan_gate_delta": combined_gate - semantic_gate,
        }
        return combined_gate

    def forward(
        self,
        in_word=None,
        feat_clip_text=None,
        emotion=None,
        mask=None,
        is_test=None,
        epoch=121,
        use_attentions=True,
        use_word=True,
        in_id=None,
        in_motion=None,
        hubert=None,
        latent=None,
    ):
        """Predict holistic motion tokens conditioned on speech and a future plan."""
        del is_test, epoch, latent  # Compatibility-only arguments.
        if hubert is None or in_word is None or in_motion is None or in_id is None:
            raise ValueError("in_word, hubert, in_motion, and in_id are required")
        if feat_clip_text is None or emotion is None or mask is None:
            raise ValueError("text features, emotion, and mask are required")

        # Fuse lexical, utterance-level and acoustic speech representations.
        in_bert_body = self.text_encoder_body(self.text_pre_encoder_body(in_word))
        in_word_body = self.hubert_encoder_body(hubert.permute(0, 2, 1)).permute(0, 2, 1)
        batch_size, frames, channels = in_bert_body.shape
        if frames % 4 != 0:
            raise ValueError("PlanTalk intent planning requires a frame length divisible by 4")

        clip_text = feat_clip_text.unsqueeze(1).expand(-1, frames, -1)
        clip_body = self.clip_embedding(clip_text)
        emotion_text = emotion.unsqueeze(1).expand(-1, frames, -1)
        emotion_body = self.emotion_embedding(emotion_text)

        semantic_attention = torch.cat([clip_body, emotion_body], dim=-1)
        semantic_attention = self.at_attn_body_semantic(semantic_attention)
        semantic_attention = semantic_attention.reshape(batch_size, frames, channels, 2).softmax(dim=-1)
        # The text embedding is used as the utterance-level semantic memory.
        fusion_body_semantic = clip_body

        bert_body = self.semantic_position_embeddings(in_bert_body)
        body_semantic = self.semantic_body_decoder(
            tgt=bert_body,
            memory=fusion_body_semantic,
        )
        fusion_attention = torch.cat([body_semantic, in_word_body], dim=-1)
        fusion_attention = self.at_attn_bert(fusion_attention)
        fusion_attention = fusion_attention.reshape(batch_size, frames, channels, 2).softmax(dim=-1)
        fusion_bert = (
            body_semantic * fusion_attention[:, :, :, 1]
            + in_word_body * fusion_attention[:, :, :, 0]
        )

        # [B, 64, 256] -> [B, 16, 256], exactly aligned with RVQ token time.
        semantic_tokens = fusion_bert.reshape(batch_size, frames // 4, 4, channels).mean(dim=2)
        plan_feature, plan_token_logits = self._predict_plan(semantic_tokens)

        # Gate frame-level semantics with plan-conditioned event timing.
        gate = self._predict_gate(semantic_tokens, plan_feature)
        body_gate = torch.softmax(gate, dim=-1)[:, :, 1]
        body_gate_frames = body_gate.unsqueeze(-1).repeat(1, 1, 4).reshape(batch_size, frames, 1)
        body_semantic = self.body_semantic_mlp(fusion_bert) * body_gate_frames

        masked_embeddings = self.mask_embeddings.expand_as(in_motion)
        masked_motion = torch.where(mask == 1, masked_embeddings, in_motion)
        body_hint = self.motion_encoder(masked_motion)
        speaker_embedding = self.spearker_encoder_body(in_id).squeeze(2)

        body_hint = self.bodyhints_body(body_hint)
        motion_embeddings = self.feature2motion(body_hint) + speaker_embedding
        motion_embeddings = self.position_embeddings(motion_embeddings)
        motion_features = self.motion_self_encoder(motion_embeddings)

        if use_word:
            semantic_memory = self.audio_feature2motion(body_semantic)
            # Speaker identity is already present in ``motion_features``.
            word_query = self.position_embeddings(motion_features)
            motion_features = motion_features + self.wordhints_decoder(
                tgt=word_query,
                memory=semantic_memory,
            )

        # First plan injection: affects the shared temporal body representation.
        motion_features = self.frame_plan_film(motion_features, plan_feature)
        motion_features = self.body1d(motion_features.permute(0, 2, 1)).permute(0, 2, 1)
        speaker_embedding = self.spearker_encoder_body1d(
            speaker_embedding.permute(0, 2, 1)
        ).permute(0, 2, 1)

        upper_hidden = self.motion2latent_upper(motion_features)
        hands_hidden = self.motion2latent_hands(motion_features)
        lower_hidden = self.motion2latent_lower(motion_features)

        # Second injection: each body branch receives an independently learned
        # plan modulation before cross-part coordination.
        upper_query = self.position_embeddings(
            self.upper_plan_film(upper_hidden, plan_feature) + speaker_embedding
        )
        hands_query = self.position_embeddings(
            self.hands_plan_film(hands_hidden, plan_feature) + speaker_embedding
        )
        lower_query = self.position_embeddings(
            self.lower_plan_film(lower_hidden, plan_feature) + speaker_embedding
        )

        motion_upper = self.upper_decoder(tgt=upper_query, memory=hands_hidden + lower_hidden)
        motion_hands = self.hands_decoder(tgt=hands_query, memory=upper_hidden + lower_hidden)
        motion_lower = self.lower_decoder(tgt=lower_query, memory=upper_hidden + hands_hidden)
        upper_latent = self.motion_down_upper(motion_upper + upper_hidden)
        hands_latent = self.motion_down_hands(motion_hands + hands_hidden)
        lower_latent = self.motion_down_lower(motion_lower + lower_hidden)

        upper_latent = self.upper1d(upper_latent.permute(0, 2, 1)).permute(0, 2, 1)
        hands_latent = self.hands1d(hands_latent.permute(0, 2, 1)).permute(0, 2, 1)
        lower_latent = self.lower1d(lower_latent.permute(0, 2, 1)).permute(0, 2, 1)

        # Third injection: plan is now exactly at the 16-token RVQ resolution.
        upper_latent = self.upper_token_plan_film(upper_latent, plan_feature)
        hands_latent = self.hands_token_plan_film(hands_latent, plan_feature)
        lower_latent = self.lower_token_plan_film(lower_latent, plan_feature)

        hands_latent = self.hands_face_decoder(
            tgt=hands_latent,
            memory=upper_latent + lower_latent,
        )
        upper_latent = self.upper_hands_decoder(
            tgt=upper_latent,
            memory=hands_latent + lower_latent,
        )
        lower_latent = self.lower_hands_decoder(
            tgt=lower_latent,
            memory=upper_latent + hands_latent,
        )

        lower_index0 = self.lower_classifier(lower_latent)
        lower_residuals = self.predict_res_lower(lower_latent, body_semantic)
        upper_index0 = self.upper_classifier(upper_latent)
        upper_residuals = self.predict_res_upper(upper_latent, body_semantic)
        hands_index0 = self.hands_classifier(hands_latent)
        hands_residuals = self.predict_res_hands(hands_latent, body_semantic)

        # ``predict_residual_zq`` returns five residual latents followed by
        # five corresponding token logits.
        lower_latent_levels = lower_residuals[:5]
        lower_index_levels = lower_residuals[5:]
        upper_latent_levels = upper_residuals[:5]
        upper_index_levels = upper_residuals[5:]
        hands_latent_levels = hands_residuals[:5]
        hands_index_levels = hands_residuals[5:]

        cls_upper = torch.stack((upper_index0, *upper_index_levels), dim=-1)
        cls_lower = torch.stack((lower_index0, *lower_index_levels), dim=-1)
        cls_hands = torch.stack((hands_index0, *hands_index_levels), dim=-1)
        rec_upper = torch.stack((upper_latent, *upper_latent_levels), dim=1).unsqueeze(2)
        rec_lower = torch.stack((lower_latent, *lower_latent_levels), dim=1).unsqueeze(2)
        rec_hands = torch.stack((hands_latent, *hands_latent_levels), dim=1).unsqueeze(2)

        outputs = {
            "gate": gate,
            "rec_upper": rec_upper,
            "rec_lower": rec_lower,
            "rec_hands": rec_hands,
            "cls_upper": cls_upper,
            "cls_lower": cls_lower,
            "cls_hands": cls_hands,
            "plan_feature": plan_feature,
            "plan_token_logits": plan_token_logits,
        }
        outputs.update(self._last_plan_diagnostics)
        outputs.update(self._last_gate_diagnostics)
        return outputs
