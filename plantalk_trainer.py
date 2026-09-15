"""Training objective for future-grounded PlanTalk generation."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from generation_trainer import GenerationTrainer
from models.future_plan_tokenizer import FuturePlanTokenizer
from utils import other_tools


class CustomTrainer(GenerationTrainer):
    """Optimize holistic generation, discrete planning, and event timing."""

    def __init__(self, args):
        super().__init__(args)
        self.lambda_plan = float(getattr(args, "lambda_plan", 0.2))
        self.lambda_plan_align = float(getattr(args, "lambda_plan_align", 0.1))
        self.plan_warmup_epochs = int(getattr(args, "plan_warmup_epochs", 10))
        self.plan_ramp_epochs = int(getattr(args, "plan_ramp_epochs", 10))
        self.plan_semantic_boost = float(getattr(args, "plan_semantic_boost", 1.0))
        self.plan_event_weight = float(getattr(args, "plan_event_weight", 0.5))
        self.plan_contrastive_weight = float(
            getattr(args, "plan_contrastive_weight", 0.1)
        )
        self.plan_contrastive_temperature = float(
            getattr(args, "plan_contrastive_temperature", 0.1)
        )
        if min(
            self.lambda_plan,
            self.lambda_plan_align,
            self.plan_event_weight,
            self.plan_contrastive_weight,
        ) < 0.0:
            raise ValueError("plan loss weights must be non-negative")
        if self.plan_contrastive_temperature <= 0.0:
            raise ValueError("plan_contrastive_temperature must be positive")

        checkpoint_path = str(getattr(args, "plan_tokenizer_ckpt", ""))
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "A pretrained plan_tokenizer_ckpt is required. Train "
                "configs/future_plan_tokenizer.yaml first."
            )
        self.plan_tokenizer = FuturePlanTokenizer(args).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state", checkpoint)
        state_dict = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
        self.plan_tokenizer.load_state_dict(state_dict, strict=True)
        self.plan_tokenizer.eval()
        for parameter in self.plan_tokenizer.parameters():
            parameter.requires_grad_(False)

        model = self.model.module if hasattr(self.model, "module") else self.model
        model.set_motion_plan_codebook(self.plan_tokenizer.quantizer.codebook)
        self._load_generator_initialization(
            model, str(getattr(args, "generator_init_ckpt", ""))
        )
        model.set_motion_plan_codebook(self.plan_tokenizer.quantizer.codebook)

        for name, higher_is_better in (
            ("plan", False),
            ("plan_acc", True),
            ("plan_align", False),
            ("plan_contrastive", False),
            ("plan_entropy", False),
            ("plan_perplexity", True),
            ("plan_active_ratio", True),
            ("plan_confidence", True),
            ("future_plan_valid_ratio", True),
            ("event", False),
            ("event_acc", True),
            ("gate_plan_shift", False),
        ):
            self._register_metric(name, higher_is_better)

    @staticmethod
    def _load_generator_initialization(model, checkpoint_path: str) -> None:
        """Optionally load shape-compatible generator parameters."""
        if not checkpoint_path:
            return
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"generator_init_ckpt does not exist: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state", checkpoint)
        state_dict = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in current
            and current[key].shape == value.shape
            and key != "motion_plan_codebook"
        }
        if not compatible:
            raise RuntimeError(
                "generator_init_ckpt contains no shape-compatible parameters"
            )
        model.load_state_dict(compatible, strict=False)

    def _register_metric(self, name: str, is_higher_better: bool) -> None:
        if name in self.tracker.metric_names:
            return
        self.tracker.metric_names.append(name)
        self.tracker.is_higher_better[name] = is_higher_better
        self.tracker.loss_meters[name] = {
            state: other_tools.AverageMeter(f"{name}_{state}")
            for state in self.tracker.states
        }
        self.tracker.values[name] = {
            state: {
                kind: {
                    "value": -np.inf if is_higher_better else np.inf,
                    "epoch": 0,
                }
                for kind in self.tracker.types
            }
            for state in self.tracker.states
        }
        self.tracker.train_history[name] = []
        self.tracker.val_history[name] = []

    def _plan_weight(self, epoch: int) -> float:
        """Linearly introduce planning after the motion objective stabilizes."""
        if epoch < self.plan_warmup_epochs or self.lambda_plan <= 0.0:
            return 0.0
        if self.plan_ramp_epochs <= 0:
            return self.lambda_plan
        progress = min(
            1.0,
            (epoch - self.plan_warmup_epochs + 1) / self.plan_ramp_epochs,
        )
        return self.lambda_plan * progress

    def _set_plan_enabled(self, enabled: bool) -> None:
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.set_plan_enabled(enabled)

    def _plan_supervision_loss(
        self,
        loaded_data: dict,
        epoch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, frames, _ = loaded_data["tar_pose"].shape
        mask = torch.ones(
            batch_size,
            frames,
            self.args.pose_dims + 3 + 4,
            device=self.device,
            dtype=torch.float32,
        )
        mask[:, : self.args.pre_frames] = 0.0
        outputs = self.model(
            loaded_data["in_word"],
            loaded_data["feat_clip_text"],
            loaded_data["emo_clip_text"],
            mask=mask,
            in_id=loaded_data["tar_id"],
            in_motion=loaded_data["latent_all"],
            use_attentions=True,
            use_word=True,
            hubert=loaded_data["hubert"],
            epoch=epoch,
        )
        logits = outputs["plan_token_logits"]
        with torch.no_grad():
            target, valid_mask = self.plan_tokenizer.encode_batch_with_mask(
                loaded_data
            )
            codebook = self.plan_tokenizer.quantizer.codebook
            target_feature = F.embedding(target, codebook)

        if logits.shape[:2] != target.shape or target.shape != valid_mask.shape:
            raise RuntimeError(
                "plan logits, targets, and validity mask must share [B, T]; "
                f"got {tuple(logits.shape)}, {tuple(target.shape)}, "
                f"and {tuple(valid_mask.shape)}"
            )
        valid_float = valid_mask.to(logits.dtype)
        valid_count = valid_float.sum().clamp_min(1.0)
        per_token_loss = F.cross_entropy(
            logits.transpose(1, 2), target, reduction="none"
        )
        semantic_weight = loaded_data["sem_mean"].to(
            device=per_token_loss.device,
            dtype=per_token_loss.dtype,
        )
        token_weight = (
            1.0 + self.plan_semantic_boost * semantic_weight
        ) * valid_float
        classification = (per_token_loss * token_weight).sum() / token_weight.sum().clamp_min(1.0)

        alignment_per_token = 1.0 - F.cosine_similarity(
            outputs["plan_feature"], target_feature, dim=-1
        )
        alignment = (alignment_per_token * valid_float).sum() / valid_count

        speech_context = F.normalize(
            outputs["speech_plan_context"][valid_mask], dim=-1
        )
        normalized_codebook = F.normalize(codebook.detach(), dim=-1)
        contrastive_logits = (
            speech_context @ normalized_codebook.t()
        ) / self.plan_contrastive_temperature
        contrastive = F.cross_entropy(contrastive_logits, target[valid_mask])

        event_target = (semantic_weight > 0).to(logits.dtype)
        event_logits = outputs["event_logits"]
        if event_logits.shape != event_target.shape:
            raise RuntimeError(
                "event logits and semantic event labels must share [B, T]; "
                f"got {tuple(event_logits.shape)} and {tuple(event_target.shape)}"
            )
        event_loss = F.binary_cross_entropy_with_logits(event_logits, event_target)
        event_accuracy = (
            (event_logits >= 0) == event_target.bool()
        ).to(logits.dtype).mean()
        gate_plan_shift = outputs["plan_gate_delta"].abs().mean()

        probabilities = outputs["plan_soft_probabilities"].detach()[valid_mask]
        mean_probability = probabilities.mean(dim=0)
        entropy = -(
            mean_probability * mean_probability.clamp_min(1e-8).log()
        ).sum()
        perplexity = entropy.exp()
        selected_ids = outputs["plan_token_ids"].detach()[valid_mask]
        active_ratio = (
            torch.bincount(selected_ids, minlength=logits.shape[-1]) > 0
        ).float().mean()
        confidence = probabilities.max(dim=-1).values.mean()
        accuracy = (
            (logits.argmax(dim=-1) == target).to(logits.dtype) * valid_float
        ).sum() / valid_count

        for name, value in (
            ("plan_align", alignment),
            ("plan_contrastive", contrastive),
            ("event", event_loss),
            ("event_acc", event_accuracy),
            ("gate_plan_shift", gate_plan_shift),
            ("plan_entropy", entropy),
            ("plan_perplexity", perplexity),
            ("plan_active_ratio", active_ratio),
            ("plan_confidence", confidence),
            ("future_plan_valid_ratio", valid_float.mean()),
        ):
            self.tracker.update_meter(name, "train", value.detach().item())

        total = (
            classification
            + self.lambda_plan_align * alignment
            + self.plan_contrastive_weight * contrastive
            + self.plan_event_weight * event_loss
        )
        return total, accuracy

    def _g_training(self, loaded_data, use_adv, mode="train", epoch=0):
        weight = self._plan_weight(epoch)
        self._set_plan_enabled(weight > 0.0)
        motion_loss = super()._g_training(loaded_data, use_adv, mode, epoch)
        if weight == 0.0:
            return motion_loss
        plan_loss, plan_accuracy = self._plan_supervision_loss(loaded_data, epoch)
        self.tracker.update_meter("plan", "train", plan_loss.detach().item())
        self.tracker.update_meter(
            "plan_acc", "train", plan_accuracy.detach().item()
        )
        return motion_loss + weight * plan_loss

    def test(self, epoch):
        self._set_plan_enabled(True)
        return super().test(epoch)

    def inference(self, audio_path):
        self._set_plan_enabled(True)
        return super().inference(audio_path)
