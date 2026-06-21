#!/usr/bin/env python3
"""Train a neural KV compactor with PyTorch Lightning."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from neural_kv.utils.config import load_config
from neural_kv.utils.rocm import apply_visible_device_for_idle_gpu, gpu_utilization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment/smoke_tiny_llama.yaml")
    parser.add_argument("--ckpt-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runtime = cfg.get("runtime", {})
    trainer_cfg = cfg.get("trainer", {})

    accelerator = str(trainer_cfg.get("accelerator", "auto"))
    if accelerator in {"gpu", "cuda", "auto"} and runtime.get("require_idle_gpu", False):
        usage = gpu_utilization()
        print(f"ROCm utilization before training: {usage}")
        selected = apply_visible_device_for_idle_gpu(
            preferred=int(runtime.get("preferred_gpu", 7)),
            require_zero=True,
        )
        if selected is None:
            raise SystemExit("No 0% utilization GPU is available; refusing to start training")
        trainer_cfg["devices"] = 1
        visible = os.environ["HIP_VISIBLE_DEVICES"]
        print(f"Selected idle GPU {selected}; HIP_VISIBLE_DEVICES={visible}")

    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    from neural_kv.training.callbacks import LegacyCheckpointCallback, StorageQuotaCallback
    from neural_kv.training.datamodule import NeuralKVDataModule
    from neural_kv.training.initialization import load_initial_compactor_state
    from neural_kv.training.lit_module import NeuralKVLightningModule

    seed = int(cfg.get("seed", 7))
    pl.seed_everything(seed, workers=True)

    output_dir = Path(cfg.get("output_dir", "checkpoints/smoke_lightning"))
    output_dir.mkdir(parents=True, exist_ok=True)

    training_cfg = cfg["training"]
    data_cfg = cfg["data"]
    max_steps = int(training_cfg.get("steps", trainer_cfg.get("max_steps", 1)))
    datamodule = NeuralKVDataModule(
        train_file=data_cfg["train_file"],
        validation_file=data_cfg.get("validation_file"),
        batch_size=int(training_cfg.get("batch_size", 1)),
        steps_per_epoch=max_steps,
        limit_train=data_cfg.get("limit_train"),
        limit_validation=data_cfg.get("limit_validation"),
        seed=seed,
        balanced_answer_sampling=bool(training_cfg.get("balanced_answer_sampling", False)),
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    module = NeuralKVLightningModule(
        model_name=cfg["model"]["name"],
        compactor=cfg["model"]["compactor"],
        context_length=int(cfg["model"].get("context_length", 8192)),
        learning_rate=float(training_cfg.get("learning_rate", 1e-4)),
        kl_weight=float(training_cfg.get("kl_weight", 1.0)),
        reverse_kl_weight=float(training_cfg.get("reverse_kl_weight", 0.0)),
        ce_weight=float(training_cfg.get("ce_weight", 0.1)),
        aux_letter_loss_weight=float(training_cfg.get("aux_letter_loss_weight", 0.0)),
        aux_letter_enable_thinking=bool(training_cfg.get("aux_letter_enable_thinking", False)),
        loss_mode=str(training_cfg.get("loss_mode", "token")),
        target_mode=str(training_cfg.get("target_mode", "letter")),
        use_chat_template=bool(training_cfg.get("use_chat_template", True)),
        enable_thinking=bool(training_cfg.get("enable_thinking", False)),
        eval_enable_thinking=bool(training_cfg.get("eval_enable_thinking", False)),
        score_mode=str(training_cfg.get("score_mode", "letter")),
        trainable_scope=str(training_cfg.get("trainable_scope", "all")),
        batch_size=int(training_cfg.get("batch_size", 1)),
        balanced_answer_sampling=bool(training_cfg.get("balanced_answer_sampling", False)),
        init_checkpoint=str(training_cfg.get("init_checkpoint") or ""),
        eval_max_new_tokens=int(training_cfg.get("eval_max_new_tokens", 192)),
        dtype=str(runtime.get("dtype", "bfloat16")),
        model_load_device=str(runtime.get("model_load_device", "auto")),
    )

    init_checkpoint = str(training_cfg.get("init_checkpoint") or "")
    if init_checkpoint and args.ckpt_path is None:
        module.initial_step = load_initial_compactor_state(
            compactor=module.compactor,
            checkpoint_path=init_checkpoint,
            model_name=cfg["model"]["name"],
            context_length=int(cfg["model"].get("context_length", 8192)),
            compactor_config=cfg["model"]["compactor"],
        )
        print(f"Loaded init checkpoint {init_checkpoint} at step {module.initial_step}")

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
        devices=trainer_cfg.get("devices", "auto"),
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
