"""Flat YAML configuration loader with command-line overrides."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml


RUNTIME_DEFAULTS = {
    "inference": False,
    "test_state": False,
    "train_rvq": False,
    "audio_infer_path": "",
    "load_ckpt": "",
    "notes": "",
    "run_name": None,
    "debug": False,
    "random_seed": 43,
    "activation": "tanh",
    "beat_align": False,
    "loader_workers": 0,
    "d_name": None,
    "d_lr_weight": 0.2,
    "no_adv_epoch": 999,
    "opt": "adam",
    "opt_eps": None,
    "opt_betas": [0.5, 0.999],
    "momentum": 0.8,
    "weight_decay": 0.0,
    "lr_min": 1.0e-7,
    "warmup_lr": 5.0e-4,
    "warmup_epochs": 0,
    "decay_rate": 0.3,
    "lr_policy": "step",
    "deterministic": True,
    "benchmark": True,
    "cudnn_enabled": True,
    "local_rank": 0,
    "vae_quantizer_lambda": 1.0,
    "render_video_fps": 30,
    "render_video_width": 1920,
    "render_video_height": 720,
    "render_concurrent_num": 4,
    "render_tmp_img_filetype": "bmp",
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"yes", "true", "t", "y", "1"}:
        return True
    if normalized in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def _argument_type(value):
    if isinstance(value, bool):
        return str2bool
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    return str


def parse_args(argv=None):
    """Load one flat YAML file and let explicit CLI flags override its values."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "-c", "--config", default="./configs/plantalk.yaml"
    )
    known, _ = config_parser.parse_known_args(argv)
    config_path = Path(known.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("the configuration must be a flat YAML mapping")
    nested = [key for key, value in loaded.items() if isinstance(value, dict)]
    if nested:
        raise ValueError(f"nested configuration sections are not supported: {nested}")

    values = {**RUNTIME_DEFAULTS, **loaded}
    parser = argparse.ArgumentParser(description="Train or evaluate PlanTalk")
    parser.add_argument("-c", "--config", default=str(config_path))
    for key, default in values.items():
        flags = [f"--{key}"]
        if key == "local_rank":
            flags.append("--local-rank")
        options = {"dest": key, "default": default}
        if isinstance(default, list):
            item_type = _argument_type(default[0]) if default else str
            options.update(type=item_type, nargs="*")
        elif isinstance(default, bool):
            options.update(type=str2bool, nargs="?", const=True)
        elif default is None:
            options.update(type=str)
        else:
            options.update(type=_argument_type(default))
        parser.add_argument(*flags, **options)
    args = parser.parse_args(argv)

    config_name = config_path.stem
    if args.run_name:
        args.name = args.run_name
    elif args.is_train:
        args.name = time.strftime("%m%d_%H%M%S_") + config_name
    else:
        args.name = config_name
    return args
