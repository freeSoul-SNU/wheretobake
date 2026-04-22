"""Loss functions for prompt baking baselines."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _gather_response_logits(
    logits: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    gathered: list[torch.Tensor] = []
    for batch_index in range(logits.shape[0]):
        sample_logits = logits[batch_index][response_mask[batch_index].bool()]
        if sample_logits.numel():
            gathered.append(sample_logits)
    if not gathered:
        return logits.new_zeros((0, logits.shape[-1]))
    return torch.cat(gathered, dim=0)


def align_student_teacher_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_response_mask: torch.Tensor,
    teacher_response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align teacher and student logits over response tokens."""

    student_rows: list[torch.Tensor] = []
    teacher_rows: list[torch.Tensor] = []
    for batch_index in range(student_logits.shape[0]):
        sample_student = student_logits[batch_index][student_response_mask[batch_index].bool()]
        sample_teacher = teacher_logits[batch_index][teacher_response_mask[batch_index].bool()]
        shared_length = min(sample_student.shape[0], sample_teacher.shape[0])
        if shared_length > 0:
            student_rows.append(sample_student[:shared_length])
            teacher_rows.append(sample_teacher[:shared_length])
    if not student_rows:
        empty = student_logits.new_zeros((0, student_logits.shape[-1]))
        return empty, empty
    return torch.cat(student_rows, dim=0), torch.cat(teacher_rows, dim=0)


def compute_token_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_response_mask: torch.Tensor,
    teacher_response_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Token-level KL on teacher-forced response positions."""

    aligned_student, aligned_teacher = align_student_teacher_logits(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_response_mask=student_response_mask,
        teacher_response_mask=teacher_response_mask,
    )
    if aligned_student.numel() == 0:
        return student_logits.new_tensor(0.0)
    teacher_probs = F.softmax(aligned_teacher / temperature, dim=-1)
    student_log_probs = F.log_softmax(aligned_student / temperature, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return loss * (temperature**2)


def compute_token_metrics(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_response_mask: torch.Tensor,
    teacher_response_mask: torch.Tensor,
) -> dict[str, Any]:
    """Teacher fidelity metrics on response tokens."""

    aligned_student, aligned_teacher = align_student_teacher_logits(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_response_mask=student_response_mask,
        teacher_response_mask=teacher_response_mask,
    )
    if aligned_student.numel() == 0:
        return {"next_token_agreement": 0.0}
    student_tokens = aligned_student.argmax(dim=-1)
    teacher_tokens = aligned_teacher.argmax(dim=-1)
    agreement = (student_tokens == teacher_tokens).float().mean().item()
    return {"next_token_agreement": agreement}

