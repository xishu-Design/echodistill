"""
Patch for distill_kl_loss to fix leaf variable in-place operation error.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def distill_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL divergence loss from student to teacher.
    KL(student || teacher) = sum(student_log_prob * log(student_prob / teacher_prob))

    This is the reverse KL (teacher as target), which is standard for knowledge distillation.
    """
    s = student_logits[:, :-1, :]
    t = teacher_logits[:, :-1, :]
    # Detach to avoid leaf variable in-place operation issues
    m = response_mask[:, 1:].detach()

    # Debug: check input shapes and values
    if s.shape != t.shape:
        raise ValueError(f"Shape mismatch: s.shape={s.shape}, t.shape={t.shape}")

    # Get log probabilities
    s_logp = F.log_softmax(s, dim=-1)
    t_logp = F.log_softmax(t, dim=-1)

    # KL divergence: KL(student || teacher) = sum(p_student * log(p_student / p_teacher))
    # In PyTorch: F.kl_div(input, target, log_target=True) expects both as log probabilities
    # This computes: sum(p_student * (log(p_student) - log(p_teacher)))
    kl = F.kl_div(s_logp, t_logp, reduction="none", log_target=True).sum(dim=-1)

    # Apply mask and average
    return (kl * m).sum() / m.sum().clamp(min=1)
