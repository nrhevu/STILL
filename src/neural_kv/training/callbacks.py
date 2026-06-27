"""Lightning callbacks for storage-aware and legacy-compatible training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from neural_kv.utils.storage import check_storage_quota, default_storage_roots

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:  # pragma: no cover
    pl = None  # type: ignore[assignment]


class StorageQuotaCallback(pl.Callback if pl is not None else object):
    """Check project-controlled storage after checkpoints are written."""

    def __init__(self, *, max_storage: str = "10TB") -> None:
        if pl is None:
            raise ModuleNotFoundError("Install the train extra to use Lightning callbacks")
        self.max_storage = max_storage

    def on_save_checkpoint(self, trainer: Any, pl_module: Any, checkpoint: dict[str, Any]) -> None:
        del trainer, pl_module, checkpoint
        check_storage_quota(default_storage_roots(), self.max_storage)


class LegacyCheckpointCallback(pl.Callback if pl is not None else object):
    """Write legacy ``train_still.py``-style ``.pt`` checkpoints from Lightning."""

    def __init__(self, *, output_dir: str | Path, save_every: int = 0) -> None:
        if pl is None:
            raise ModuleNotFoundError("Install the train extra to use Lightning callbacks")
        self.output_dir = Path(output_dir)
        self.save_every = int(save_every)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"

    @staticmethod
    def _metrics(trainer: Any) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                metrics[str(key)] = float(value.detach().cpu())
            elif isinstance(value, int | float):
                metrics[str(key)] = float(value)
        return metrics

    @staticmethod
    def _checkpoint_payload(
        pl_module: Any,
        *,
        step: int,
        metrics: dict[str, float],
    ) -> dict[str, Any]:
        compactor_cfg = dict(pl_module.hparams.compactor)
        hparams = pl_module.hparams
        return {
            "step": step,
            "model": str(hparams.model_name),
            "num_latents": int(compactor_cfg["num_latents"]),
            "sink_tokens": int(compactor_cfg.get("sink_tokens", 0)),
            "exact_tokens": int(compactor_cfg.get("exact_tokens", 0)),
            "exact_strategy": str(compactor_cfg.get("exact_strategy", "prefix")),
            "exact_beta": float(compactor_cfg.get("exact_beta", 0.0)),
            "num_blocks": int(compactor_cfg.get("num_blocks", 2)),
            "layer_compactor_groups": int(compactor_cfg.get("layer_compactor_groups", 0)),
            "head_specific_latents": bool(compactor_cfg.get("head_specific_latents", False)),
            "rope_mode": str(compactor_cfg.get("rope_mode", "default")),
            "beta_base": str(compactor_cfg.get("beta_base", "zero")),
            "beta_init": float(compactor_cfg.get("beta_init", 0.0)),
            "context_length": int(hparams.context_length),
            "batch_size": int(getattr(hparams, "batch_size", 1)),
            "kl_weight": float(getattr(hparams, "kl_weight", 1.0)),
            "reverse_kl_weight": float(getattr(hparams, "reverse_kl_weight", 0.0)),
            "ce_weight": float(getattr(hparams, "ce_weight", 0.0)),
            "aux_letter_loss_weight": float(getattr(hparams, "aux_letter_loss_weight", 0.0)),
            "aux_letter_enable_thinking": bool(
                getattr(hparams, "aux_letter_enable_thinking", False)
            ),
            "latent_dropout": float(compactor_cfg.get("latent_dropout", 0.0)),
            "loss_mode": str(getattr(hparams, "loss_mode", "letter")),
            "target_mode": str(getattr(hparams, "target_mode", "letter")),
            "score_mode": str(getattr(hparams, "score_mode", "letter_logprob")),
            "eval_max_new_tokens": int(getattr(hparams, "eval_max_new_tokens", 192)),
            "init_checkpoint": str(getattr(hparams, "init_checkpoint", "") or ""),
            "use_chat_template": bool(getattr(hparams, "use_chat_template", True)),
            "enable_thinking": bool(getattr(hparams, "enable_thinking", False)),
            "eval_enable_thinking": bool(getattr(hparams, "eval_enable_thinking", False)),
            "balanced_answer_sampling": bool(getattr(hparams, "balanced_answer_sampling", False)),
            "trainable_scope": str(getattr(hparams, "trainable_scope", "all")),
            "state_dict": pl_module.compactor.state_dict(),
            "metrics": metrics,
        }

    def _legacy_step(self, trainer: Any, pl_module: Any) -> int:
        return int(getattr(pl_module, "initial_step", 0)) + int(trainer.global_step)

    def _save(self, path: Path, trainer: Any, pl_module: Any) -> None:
        metrics = self._metrics(trainer)
        step = self._legacy_step(trainer, pl_module)
        payload = self._checkpoint_payload(pl_module, step=step, metrics=metrics)
        torch.save(payload, path)
        metrics_row = {"step": float(step), **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics_row, sort_keys=True) + "\n")

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        if self.save_every <= 0 or trainer.global_step <= 0:
            return
        if self._legacy_step(trainer, pl_module) % self.save_every == 0:
            path = self.output_dir / f"step_{self._legacy_step(trainer, pl_module)}.pt"
            self._save(path, trainer, pl_module)

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        self._save(self.output_dir / "final.pt", trainer, pl_module)
        metrics = self._metrics(trainer)
        with (self.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
