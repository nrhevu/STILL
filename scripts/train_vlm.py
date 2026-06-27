#!/usr/bin/env python3
"""Train a VLM neural KV compactor after full-cache gate checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_kv.utils.config import load_config
from neural_kv.utils.rocm import ensure_last_four_gpu_visibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/experiment/vlm_qwen3_vl_8b_scienceqa_4gpu.yaml",
    )
    parser.add_argument("--ckpt-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visible = ensure_last_four_gpu_visibility()
    print(f"HIP_VISIBLE_DEVICES={visible}")

    cfg = load_config(args.config)
    gates = cfg.get("gates", [])
    if not gates:
        raise SystemExit("VLM training config must declare full-cache gates")
    from neural_kv.training.vlm import check_gate_summaries

    check_gate_summaries(list(gates))

    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    from neural_kv.training.callbacks import LegacyCheckpointCallback, StorageQuotaCallback
    from neural_kv.training.datamodule import NeuralKVDataModule
    from neural_kv.training.vlm import VLMNeuralKVLightningModule

    seed = int(cfg.get("seed", 7))
    pl.seed_everything(seed, workers=True)

    output_dir = Path(cfg.get("output_dir", "checkpoints/vlm_qwen3_vl_8b_scienceqa"))
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = cfg.get("runtime", {})
    trainer_cfg = cfg.get("trainer", {})
    training_cfg = cfg["training"]
    data_cfg = cfg["data"]
    max_steps = int(training_cfg.get("steps", trainer_cfg.get("max_steps", 1)))
    devices = trainer_cfg.get("devices", "auto")
    sampler_steps = max_steps
    if isinstance(devices, int) and devices > 1:
        sampler_steps = max_steps * devices

    datamodule = NeuralKVDataModule(
        train_file=data_cfg["train_file"],
        validation_file=data_cfg.get("validation_file"),
        batch_size=int(training_cfg.get("batch_size", 1)),
        steps_per_epoch=sampler_steps,
        limit_train=data_cfg.get("limit_train"),
        limit_validation=data_cfg.get("limit_validation"),
        seed=seed,
        balanced_answer_sampling=bool(training_cfg.get("balanced_answer_sampling", False)),
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    module = VLMNeuralKVLightningModule(
        model_name=cfg["model"]["name"],
        compactor=cfg["model"]["compactor"],
        context_length=int(cfg["model"].get("context_length", 8192)),
        train_base_dir=str(Path(data_cfg["train_file"]).parent),
        validation_base_dir=str(
            Path(data_cfg.get("validation_file", data_cfg["train_file"])).parent
        ),
        learning_rate=float(training_cfg.get("learning_rate", 5e-6)),
        kl_weight=float(training_cfg.get("kl_weight", 1.0)),
        ce_weight=float(training_cfg.get("ce_weight", 1.0)),
        dtype=str(runtime.get("dtype", "bfloat16")),
        model_load_device=str(runtime.get("model_load_device", "auto")),
    )

    callbacks = [
        LegacyCheckpointCallback(
            output_dir=output_dir,
            save_every=int(training_cfg.get("save_every", 0)),
        ),
        ModelCheckpoint(
            dirpath=output_dir,
            filename="step_{step}",
            every_n_train_steps=int(training_cfg.get("save_every", 0)) or None,
            save_last=True,
        ),
        StorageQuotaCallback(max_storage=str(runtime.get("max_storage", "10TB"))),
    ]
    logger = CSVLogger(save_dir=str(output_dir), name="logs")
    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=devices,
        strategy=trainer_cfg.get("strategy", "auto"),
        precision=trainer_cfg.get("precision", "bf16-mixed"),
        max_steps=max_steps,
        num_sanity_val_steps=0,
        gradient_clip_val=float(training_cfg.get("gradient_clip_val", 1.0)),
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 1)),
        val_check_interval=trainer_cfg.get("val_check_interval"),
        check_val_every_n_epoch=trainer_cfg.get("check_val_every_n_epoch", 1),
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
