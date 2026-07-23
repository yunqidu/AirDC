#!/usr/bin/env python3
"""Convert old checkpoints to the cleaned public AirDC module layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


KEY_RENAME_PREFIXES = (
    ("feature.", "shared_feature_extractor."),
    ("lift_feature.", "sd_fpnet."),
    ("cost_agg.", "asr_net."),
    ("cnet.", "context_extractor."),
    ("update_block.", "ihcfr."),
    ("context_zqr_convs.", "ihcfr_context_projections."),
    ("cam.", "ihcfr_channel_attention."),
    ("sam.", "ihcfr_spatial_attention."),
)


def resolve_path(path: str | None, base: Path) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def load_config_dict(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_state_dict(checkpoint_obj: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint_obj and all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return checkpoint_obj
    raise ValueError("Could not find a valid state_dict in the input checkpoint.")


def normalize_key(key: str, strip_module_prefix: bool, rename_public_keys: bool) -> str:
    if strip_module_prefix and key.startswith("module."):
        key = key[len("module."):]
    if rename_public_keys:
        for old_prefix, new_prefix in KEY_RENAME_PREFIXES:
            if key.startswith(old_prefix):
                return new_prefix + key[len(old_prefix):]
    return key


def convert_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
    strip_module_prefix: bool,
    rename_public_keys: bool,
) -> dict[str, torch.Tensor]:
    converted = {}
    for key, value in state_dict.items():
        new_key = normalize_key(key, strip_module_prefix, rename_public_keys)
        if new_key in converted:
            raise ValueError(f"Duplicate key after conversion: {new_key}")
        converted[new_key] = value
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Repack an AirDC checkpoint into a clean public checkpoint.")
    parser.add_argument("--config", default="config/val_all_att.yaml", help="AirDC config path.")
    parser.add_argument("--workspace", default=".", help="Repository/workspace root.")
    parser.add_argument("--log-dir", default="log", help="Log directory relative to workspace.")
    parser.add_argument("--input-ckpt", default=None, help="Old checkpoint path. Defaults to ckpt_path in config.")
    parser.add_argument("--output-ckpt", required=True, help="Converted checkpoint output path.")
    parser.add_argument("--strip-module-prefix", action="store_true", help="Strip leading 'module.' from all state_dict keys.")
    parser.add_argument(
        "--no-rename-public-keys",
        action="store_true",
        help="Keep legacy module key names instead of converting them to public AirDC names.",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    config_path = resolve_path(args.config, workspace)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    config_dict = load_config_dict(config_path)
    input_ckpt = resolve_path(args.input_ckpt, workspace)
    if input_ckpt is None:
        ckpt_paths = config_dict.get("ckpt_path", [])
        ckpts = ckpt_paths if isinstance(ckpt_paths, list) else [ckpt_paths]
        base_log_dir = workspace / args.log_dir
        ckpts = [str(resolve_path(ckpt, base_log_dir)) if ckpt else "" for ckpt in ckpts]
        input_ckpt = Path(next((ckpt for ckpt in ckpts if ckpt), ""))
    if not input_ckpt.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_ckpt}")

    old_ckpt = torch.load(input_ckpt, map_location="cpu")
    state_dict = extract_state_dict(old_ckpt)
    state_dict = convert_state_dict_keys(
        state_dict,
        strip_module_prefix=args.strip_module_prefix,
        rename_public_keys=not args.no_rename_public_keys,
    )
    output_ckpt = resolve_path(args.output_ckpt, workspace)
    assert output_ckpt is not None
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)

    converted = {
        "model_name": "AirDC",
        "model_state_dict": state_dict,
        "converted_from": str(input_ckpt),
        "config_path": str(config_path),
    }
    torch.save(converted, output_ckpt)
    print(f"Converted AirDC checkpoint saved to: {output_ckpt}")


if __name__ == "__main__":
    main()
