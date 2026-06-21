"""PyTorch Lightning module for neural KV distillation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from neural_kv.models.compactor import StillCompactor
from neural_kv.modules.attention_bias import enable_still_attention_bias
from neural_kv.training.distillation import (
    dtype_from_name,
    load_model_and_tokenizer,
    resolve_device,
    score_mcq_letters,
    score_mcq_no_context,
    training_forward,
)

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:  # pragma: no cover - exercised only without train extra
    pl = None  # type: ignore[assignment]


TRAINABLE_SCOPES: dict[str, tuple[str, ...] | None] = {
    "all": None,
    "beta": ("beta_head",),
    "heads": ("key_head", "value_head"),
    "beta_heads": ("beta_head", "key_head", "value_head"),
    "latents": ("latents",),
    "latents_beta": ("latents", "beta_head"),
}


def _device_name(device: torch.device) -> str:
    if device.type == "cuda" and device.index is not None:
        return f"cuda:{device.index}"
    return str(device)


class NeuralKVLightningModule(pl.LightningModule if pl is not None else nn.Module):
    """Lightning orchestration around the plain PyTorch STILL compactor."""

    def __init__(
        self,
        *,
        model_name: str,
        compactor: dict[str, Any],
        context_length: int,
        learning_rate: float = 1e-4,
        kl_weight: float = 1.0,
        reverse_kl_weight: float = 0.0,
        ce_weight: float = 0.1,
        aux_letter_loss_weight: float = 0.0,
        aux_letter_enable_thinking: bool = False,
        loss_mode: str = "token",
        target_mode: str = "letter",
        use_chat_template: bool = True,
        enable_thinking: bool = False,
        eval_enable_thinking: bool = False,
        score_mode: str = "letter",
        trainable_scope: str = "all",
        batch_size: int = 1,
        balanced_answer_sampling: bool = False,
        init_checkpoint: str = "",
        eval_max_new_tokens: int = 192,
        dtype: str = "bfloat16",
        model_load_device: str = "auto",
    ) -> None:
        if pl is None:
            raise ModuleNotFoundError("Install the train extra to use NeuralKVLightningModule")
        super().__init__()
        if trainable_scope not in TRAINABLE_SCOPES:
            raise ValueError(f"Unsupported trainable_scope: {trainable_scope}")
        self.save_hyperparameters()

        load_device = resolve_device(model_load_device)
        base_model, tokenizer = load_model_and_tokenizer(
            model_name,
            device=load_device,
            dtype=dtype_from_name(dtype),
        )
        object.__setattr__(self, "base_model", base_model)
        object.__setattr__(self, "tokenizer", tokenizer)
        self.compactor = StillCompactor.from_model_config(base_model.config, **compactor)
        self.initial_step = 0
        self._patched_attention_layers = 0
        self._apply_trainable_scope()

    @property
    def patched_attention_layers(self) -> int:
        return int(self._patched_attention_layers)

    def _apply_trainable_scope(self) -> None:
        keywords = TRAINABLE_SCOPES[str(self.hparams.trainable_scope)]
        trainable: list[nn.Parameter] = []
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.compactor.named_parameters():
            enabled = keywords is None or any(keyword in name for keyword in keywords)
            parameter.requires_grad_(enabled)
            if enabled:
                trainable.append(parameter)
        if not trainable:
            raise ValueError(f"No trainable parameters selected for {self.hparams.trainable_scope}")

    def on_fit_start(self) -> None:
        self.base_model.to(self.device)
        self.base_model.eval()
        self._patched_attention_layers = enable_still_attention_bias(self.base_model)
        self.compactor.train()

    def on_validation_start(self) -> None:
        self.base_model.to(self.device)
        self.base_model.eval()
        if not self._patched_attention_layers:
            self._patched_attention_layers = enable_still_attention_bias(self.base_model)

    def forward(self, past_key_values, **kwargs):
        return self.compactor(past_key_values, **kwargs)

    def training_step(self, batch: list[dict[str, Any]], batch_idx: int) -> torch.Tensor:
        del batch_idx
        device = _device_name(self.device)
        losses: list[torch.Tensor] = []
        metric_sums: dict[str, float] = {}
        for row in batch:
            primary_loss, metrics = training_forward(
                model=self.base_model,
                tokenizer=self.tokenizer,
                compactor=self.compactor,
                row=row,
                context_length=int(self.hparams.context_length),
                device=device,
                kl_weight=float(self.hparams.kl_weight),
                ce_weight=float(self.hparams.ce_weight),
                reverse_kl_weight=float(self.hparams.reverse_kl_weight),
                target_mode=str(self.hparams.target_mode),
                loss_mode=str(self.hparams.loss_mode),
                use_chat_template=bool(self.hparams.use_chat_template),
                enable_thinking=bool(self.hparams.enable_thinking),
            )
            loss = primary_loss
            metrics["primary_loss"] = float(primary_loss.detach().cpu())
            if float(self.hparams.aux_letter_loss_weight) > 0:
                aux_loss, aux_metrics = training_forward(
                    model=self.base_model,
                    tokenizer=self.tokenizer,
                    compactor=self.compactor,
                    row=row,
                    context_length=int(self.hparams.context_length),
                    device=device,
                    kl_weight=float(self.hparams.kl_weight),
                    ce_weight=float(self.hparams.ce_weight),
                    reverse_kl_weight=float(self.hparams.reverse_kl_weight),
                    target_mode="letter",
                    loss_mode="letter",
                    use_chat_template=bool(self.hparams.use_chat_template),
                    enable_thinking=bool(self.hparams.aux_letter_enable_thinking),
                )
                loss = loss + float(self.hparams.aux_letter_loss_weight) * aux_loss
                metrics["aux_letter_loss"] = float(aux_loss.detach().cpu())
                for key, value in aux_metrics.items():
                    metrics[f"aux_letter_{key}"] = value
            losses.append(loss)
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

        batch_size = max(len(batch), 1)
        mean_loss = torch.stack(losses).mean()
        self.log("train/loss", mean_loss, prog_bar=True, on_step=True, on_epoch=False)
        for key, value in metric_sums.items():
            self.log(f"train/{key}", value / batch_size, on_step=True, on_epoch=False)
        return mean_loss

    def validation_step(self, batch: list[dict[str, Any]], batch_idx: int) -> None:
        del batch_idx
        if not batch:
            return
        row = batch[0]
        device = _device_name(self.device)
        gold = row.get("answer_index")
        no_context = score_mcq_no_context(
            model=self.base_model,
            tokenizer=self.tokenizer,
            row=row,
            device=device,
            score_mode=str(self.hparams.score_mode),
            use_chat_template=bool(self.hparams.use_chat_template),
            enable_thinking=bool(self.hparams.eval_enable_thinking),
        )
        full, _ = score_mcq_letters(
            model=self.base_model,
            tokenizer=self.tokenizer,
            row=row,
            context_length=int(self.hparams.context_length),
            device=device,
            compactor=None,
            score_mode=str(self.hparams.score_mode),
            use_chat_template=bool(self.hparams.use_chat_template),
            enable_thinking=bool(self.hparams.eval_enable_thinking),
        )
        compact, meta = score_mcq_letters(
            model=self.base_model,
            tokenizer=self.tokenizer,
            row=row,
            context_length=int(self.hparams.context_length),
            device=device,
            compactor=self.compactor,
            score_mode=str(self.hparams.score_mode),
            use_chat_template=bool(self.hparams.use_chat_template),
            enable_thinking=bool(self.hparams.eval_enable_thinking),
        )
        labels = "ABCD"
        target = labels[int(gold)] if isinstance(gold, int) else None
        self.log("val/no_context_accuracy", float(no_context == target), on_epoch=True)
        self.log("val/full_accuracy", float(full == target), on_epoch=True)
        self.log("val/compact_accuracy", float(compact == target), prog_bar=True, on_epoch=True)
        self.log("val/compression", float(meta.get("compression", 0.0)), on_epoch=True)

    def configure_optimizers(self):
        params = [parameter for parameter in self.compactor.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(params, lr=float(self.hparams.learning_rate))

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        metadata = checkpoint.get("neural_kv", {})
        self.initial_step = int(metadata.get("initial_step", 0))

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["neural_kv"] = {
            "model": self.hparams.model_name,
            "context_length": int(self.hparams.context_length),
            "compactor": dict(self.hparams.compactor),
            "loss_mode": self.hparams.loss_mode,
            "target_mode": self.hparams.target_mode,
            "score_mode": self.hparams.score_mode,
            "trainable_scope": self.hparams.trainable_scope,
            "initial_step": int(self.initial_step),
        }
