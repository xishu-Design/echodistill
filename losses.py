from __future__ import annotations

import torch
import torch.nn.functional as F


def grpo_policy_loss(
    student_logps: torch.Tensor,
    rewards: torch.Tensor,
    old_logps: torch.Tensor = None,
    is_clip: float = None,
    response_mask = None,
) -> torch.Tensor:
    """Compute a GRPO/PPO-style sequence-level policy loss.

    Rewards are normalized within the sampled group and treated as constants
    with respect to the policy update.
    """
    eps = 1e-6
    del response_mask

    rewards = rewards.detach()
    if rewards.numel() > 1:
        mean_reward = rewards.mean()
        std_reward = rewards.std(unbiased=False)
        advantages = (rewards - mean_reward) / std_reward.clamp(min=eps)
    else:
        advantages = rewards
    advantages = advantages.detach()

    if old_logps is None:
        old_logps = student_logps.detach()
    else:
        old_logps = old_logps.detach()

    log_ratio = student_logps - old_logps
    ratio = torch.exp(log_ratio)

    if is_clip is not None and is_clip > 0:
        clipped_ratio = torch.clamp(ratio, 1.0 - is_clip, 1.0 + is_clip)
        surrogate = ratio * advantages
        clipped_surrogate = clipped_ratio * advantages
        loss = -torch.min(surrogate, clipped_surrogate).mean()
    else:
        loss = -(ratio * advantages).mean()

    return loss


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
    m = response_mask[:, 1:]

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
