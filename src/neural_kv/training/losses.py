"""Training losses for teacher-student KV compaction."""

from neural_kv.training.distillation import kl_and_ce_loss, letter_kl_and_ce_loss

__all__ = ["kl_and_ce_loss", "letter_kl_and_ce_loss"]
