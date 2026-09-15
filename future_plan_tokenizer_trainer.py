"""Standalone pretraining loop for the future-motion plan tokenizer."""

from __future__ import annotations

import time

import torch

import train
from utils import other_tools


class CustomTrainer(train.TrainingRuntime):
    def __init__(self, args):
        super().__init__(args)
        self.tracker = other_tools.EpochTracker(
            ["plan_recon", "plan_commit", "plan_perplexity"],
            [False, False, True],
        )

    def _move_to_device(self, value):
        if torch.is_tensor(value):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, dict):
            return {key: self._move_to_device(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._move_to_device(item) for item in value)
        return value

    def train(self, epoch):
        self.model.train()
        start = time.time()
        self.tracker.reset()
        for iteration, batch in enumerate(self.train_loader):
            batch = self._move_to_device(batch)
            self.opt.zero_grad(set_to_none=True)
            outputs = self.model(batch)
            outputs["total_loss"].backward()
            if self.args.grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()

            self.tracker.update_meter("plan_recon", "train", outputs["reconstruction_loss"].detach().item())
            self.tracker.update_meter("plan_commit", "train", outputs["commitment_loss"].detach().item())
            self.tracker.update_meter("plan_perplexity", "train", outputs["perplexity"].detach().item())
            if iteration % self.args.log_period == 0:
                memory_gb = torch.cuda.memory_reserved() / 1e9
                self.train_recording(
                    epoch,
                    iteration,
                    0.0,
                    time.time() - start,
                    memory_gb,
                    self.opt.param_groups[0]["lr"],
                )
                start = time.time()
        self.opt_s.step(epoch)

    def test(self, epoch):
        # Tokenizer checkpoints are selected from reconstruction and code-use
        # statistics, so full gesture rendering is not needed at this stage.
        return None
