import torch

from neural_kv.hf_training import letter_kl_and_ce_loss


def test_letter_kl_and_ce_loss_rewards_gold_letter() -> None:
    teacher_logits = torch.tensor([[0.0, 3.0, -1.0, -2.0]])
    good_student = torch.tensor([[0.0, 2.0, -1.0, -2.0]])
    bad_student = torch.tensor([[3.0, 0.0, -1.0, -2.0]])

    good_loss = letter_kl_and_ce_loss(
        teacher_logits=teacher_logits,
        student_logits=good_student,
        target_index=1,
        kl_weight=1.0,
        ce_weight=1.0,
    )
    bad_loss = letter_kl_and_ce_loss(
        teacher_logits=teacher_logits,
        student_logits=bad_student,
        target_index=1,
        kl_weight=1.0,
        ce_weight=1.0,
    )

    assert good_loss < bad_loss
