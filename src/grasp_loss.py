"""GRASP distribution-matching loss and trainer integration.

Apply the context patch before the loss patch. LAMPQA_GRASP_LOSS enables
completion-only distillation; dataset/SFT rows retain the engine loss.
The objective follows OPSD and adapts TRL's generalized JSD.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from logging_utils import get_logger

__all__ = ["GRASPLossConfig", "grasp_jsd_loss", "forward_kl_loss", "apply_grasp_loss_patch"]

_GRASP_LOSS_ENV = "LAMPQA_GRASP_LOSS"


class GRASPLossConfig:
    """Validated divergence settings. Clipping is off by default because it changes
    the objective, rather than merely improving numerical stability."""

    def __init__(
        self,
        beta: float = 0.0,
        temperature: float = 1.0,
        top_k: int | None = None,
        token_clip: float | None = None,
    ):
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be > 0 or None, got {top_k}")
        if token_clip is not None and token_clip <= 0:
            raise ValueError(f"token_clip must be > 0 or None, got {token_clip}")
        self.beta = beta
        self.temperature = temperature
        self.top_k = top_k
        self.token_clip = token_clip

    def __repr__(self):  # pragma: no cover - debug aid
        return (
            f"GRASPLossConfig(beta={self.beta}, temperature={self.temperature}, "
            f"top_k={self.top_k}, token_clip={self.token_clip})"
        )

    @property
    def divergence_name(self) -> str:
        if self.beta == 0.0:
            return "forward_kl(p_T||p_S)"
        if self.beta == 1.0:
            return "reverse_kl(p_S||p_T)"
        return f"jsd(beta={self.beta})"


def grasp_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.0,
    temperature: float = 1.0,
    top_k: int | None = None,
    token_clip: float | None = None,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Compute masked generalized JSD, adapted from TRL 0.17.0.

    Logits have shape [B, S, V]; teacher logits must be detached by the caller.
    Labels have shape [B, S], with -100 excluding prompt and padding positions.
    beta selects forward KL (0), reverse KL (1), or JSD (between 0 and 1).
    top_k renormalizes both distributions on the teacher top-k support.
    token_clip caps each vocabulary cell, retaining negative contributions;
    the clipped loss can therefore be negative.

    batchmean divides by supervised positions; sum/mean/none are also supported.
    Returns a scalar except for reduction="none"."""
    if labels is None:
        raise ValueError(
            "labels is mandatory: with labels=None the divergence would be averaged "
            "over prompt and padding positions too. The teacher prompt is longer than "
            "the student's in this project, so those positions are not aligned and "
            "contribute noise."
        )
    if student_logits.shape[:2] != teacher_logits.shape[:2]:
        raise ValueError(
            f"student/teacher must agree on [B, S]: got {tuple(student_logits.shape)} "
            f"vs {tuple(teacher_logits.shape)}. If these differ, the completion "
            f"segments were not aligned by the caller."
        )
    if labels.shape != student_logits.shape[:2]:
        raise ValueError(
            f"labels must be [B, S] matching the logits: got {tuple(labels.shape)} "
            f"vs {tuple(student_logits.shape[:2])}"
        )

    mask = labels != -100
    n_kept = int(mask.sum())
    if n_kept == 0:
        raise ValueError(
            "every position is masked out (labels == -100 everywhere). The loss would "
            "be 0/0. This usually means the completion span was lost upstream."
        )

    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    if top_k is not None and top_k > 0 and top_k < teacher_logits.shape[-1]:
        # Restrict to the teacher's most probable entries, then renormalise both
        # sides over that shared support.
        _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
        student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
        teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        # forward KL(p_T || p_S).
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        # log1p(-beta) rather than log(1 - beta): stable as beta -> 0.
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [
                    student_log_probs + torch.log1p(-beta_t),
                    teacher_log_probs + torch.log(beta_t),
                ]
            ),
            dim=0,
        )
        kl_teacher = F.kl_div(
            mixture_log_probs, teacher_log_probs, reduction="none", log_target=True
        )
        kl_student = F.kl_div(
            mixture_log_probs, student_log_probs, reduction="none", log_target=True
        )
        jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

    if token_clip is not None:
        jsd = jsd.clamp(max=token_clip)

    if reduction == "none":
        # Zero out the masked positions rather than boolean-indexing, so the
        # caller still gets a [B, S, V] tensor it can inspect per position.
        return jsd * mask.unsqueeze(-1)

    jsd = jsd[mask]

    if reduction == "batchmean":
        # Normalize by supervised positions, without a second sequence-length division.
        return jsd.sum() / n_kept
    if reduction == "sum":
        return jsd.sum()
    if reduction == "mean":
        return jsd.mean()
    raise ValueError(f"unknown reduction: {reduction!r}")


def forward_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """``grasp_jsd_loss`` pinned to ``beta=0`` (forward KL), OPSD's main setting."""
    kwargs.pop("beta", None)
    return grasp_jsd_loss(student_logits, teacher_logits, labels, beta=0.0, **kwargs)


def _announce(msg, level="info"):
    """Keep patch diagnostics visible after framework logging setup."""
    getattr(get_logger(__name__), level)("%s", msg)


def _loss_env(name: str, default: str = "") -> str:
    """Prefer GRASP settings, falling back to legacy LAMPQA_OPSD variables."""
    legacy = name.replace("LAMPQA_GRASP_LOSS", "LAMPQA_OPSD", 1)
    return os.environ.get(name, os.environ.get(legacy, default)).strip()


def _grasp_loss_enabled() -> bool:
    return _loss_env(_GRASP_LOSS_ENV) in {"1", "true", "True", "yes"}


def config_from_env() -> GRASPLossConfig:
    """Read loss hyper-parameters from the environment."""

    def _f(name, default):
        raw = _loss_env(name)
        if not raw:
            return default
        return float(raw)

    top_k = int(_f("LAMPQA_GRASP_LOSS_TOP_K", 0))
    token_clip = _f("LAMPQA_GRASP_LOSS_TOKEN_CLIP", 0.0)
    return GRASPLossConfig(
        beta=_f("LAMPQA_GRASP_LOSS_BETA", 0.0),
        temperature=_f("LAMPQA_GRASP_LOSS_TEMPERATURE", 1.0),
        top_k=top_k if top_k > 0 else None,
        token_clip=token_clip if token_clip > 0 else None,
    )


def infer_completion_length(inputs) -> int:
    """Prefer logits_to_keep; labels may already be sliced to the completion window."""
    n_keep = inputs.get("logits_to_keep")
    if isinstance(n_keep, torch.Tensor):
        n_keep = int(n_keep.max().item())
    if n_keep:
        return int(n_keep)

    labels = inputs.get("labels")
    if labels is None:
        raise ValueError(
            "cannot determine the completion length: neither 'logits_to_keep' "
            "nor 'labels' is present in the batch"
        )
    per_row = (labels != -100).sum(dim=1)
    n = int(per_row.max().item())
    if n == 0:
        raise ValueError(
            "labels are -100 everywhere, so the completion window is empty; "
            "the rollout/collator stage is broken"
        )
    return n


def build_completion_labels(inputs, n_comp: int, device) -> torch.Tensor:
    """Return [B, n_comp] labels, right-aligned with -100 padding.
    Without labels, assume all completion positions are supervised."""
    labels = inputs.get("labels")
    if labels is None:
        b = inputs["input_ids"].shape[0]
        return torch.ones(b, n_comp, dtype=torch.long, device=device)

    if labels.shape[1] == n_comp:
        return labels.to(device)
    if labels.shape[1] > n_comp:
        return labels[:, -n_comp:].to(device)

    # labels narrower than the window: right-align it and mask the rest.
    b = labels.shape[0]
    out = torch.full((b, n_comp), -100, dtype=labels.dtype, device=device)
    out[:, -labels.shape[1] :] = labels.to(device)
    return out


def align_completion_logits(logits: torch.Tensor, n_comp: int) -> torch.Tensor:
    """Select logits predicting the trailing completion tokens.
    Full sequences use [-n_comp-1:-1] for the causal shift. A window already
    reduced to n_comp positions is assumed aligned."""
    s = logits.shape[1]
    if s == n_comp:
        # Already sliced by the engine; assumed to be the aligned window.
        return logits
    if s < n_comp:
        raise ValueError(
            f"logits sequence length {s} is shorter than the completion window "
            f"{n_comp}; alignment is impossible"
        )
    return logits[:, -n_comp - 1 : -1, :]


def apply_grasp_loss_patch():
    """Monkey-patch ``GKDTrainer.compute_loss`` to use the GRASP loss."""
    from swift.trainers.rlhf_trainer.gkd_trainer import DataSource, GKDTrainer

    if not _grasp_loss_enabled():
        _announce(f"[GRASP] {_GRASP_LOSS_ENV} not set -> leaving compute_loss unpatched")
        return

    if getattr(GKDTrainer, "_grasp_loss_patched", False):
        _announce("[GRASP] already patched")
        return

    cfg = config_from_env()
    _announce(f"[GRASP] patching compute_loss with {cfg} ({cfg.divergence_name})")

    orig_compute_loss = GKDTrainer.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        data_source = inputs.get("_data_source", DataSource.DATASET)

        # Only student rollouts get the on-policy objective. Dataset rows (plain
        # SFT) keep the engine's own path.
        if data_source != DataSource.STUDENT:
            return orig_compute_loss(
                self,
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        # The teacher-input builder is installed on the trainer; access its bound method.
        from grasp_patch import _TEACHER_PROMPT_KEY

        build_teacher_inputs = getattr(self, "_grasp_build_teacher_inputs", None)

        teacher_prompt_ids = inputs.get(_TEACHER_PROMPT_KEY)
        if teacher_prompt_ids is None:
            teacher_prompt_ids = getattr(self, "_grasp_active_prompt_ids", None)

        n_comp = infer_completion_length(inputs)

        student_inputs = {
            k: v
            for k, v in inputs.items()
            if k
            not in {
                "prompt",
                "labels",
                "_data_source",
                "old_per_token_logps",
                "teacher_per_token_logps",
                _TEACHER_PROMPT_KEY,
            }
        }
        # Ask for full-vocab logits over the completion window only; this keeps
        # the [B, S, V] intermediate from spanning the (long) profile prompt.
        student_inputs["logits_to_keep"] = n_comp + 1

        outputs_student = model(**student_inputs)
        student_logits = align_completion_logits(outputs_student.logits, n_comp)

        # --- teacher forward on its privileged context ------------------------
        teacher_inputs = dict(student_inputs)
        if teacher_prompt_ids is not None:
            probe = dict(inputs)
            probe["logits_to_keep"] = n_comp
            if build_teacher_inputs is None:
                raise RuntimeError(
                    "GKDTrainer has no _grasp_build_teacher_inputs: the GRASP patch is "
                    "not installed, so there is no way to give the teacher its "
                    "privileged context. Set LAMPQA_GRASP=1 and ensure "
                    "apply_grasp_patch() runs before apply_grasp_loss_patch()."
                )
            # Bound method: pass (inputs, teacher_prompts), not (self, ...).
            teacher_inputs = build_teacher_inputs(probe, teacher_prompt_ids)
            teacher_inputs = {
                k: v
                for k, v in teacher_inputs.items()
                if k not in {"prompt", "labels", "_data_source", _TEACHER_PROMPT_KEY}
            }
            teacher_inputs["logits_to_keep"] = n_comp + 1
        elif os.environ.get("LAMPQA_GRASP", "").strip() in {"1", "true", "True", "yes"}:
            raise RuntimeError(
                "LAMPQA_GRASP is set but no privileged teacher prompt reached "
                "compute_loss; the teacher would score the student's own context "
                "and the divergence would be zero by construction."
            )

        load_context = (
            self.load_teacher_model_context()
            if getattr(self.args, "offload_teacher_model", False)
            else nullcontext()
        )
        with torch.no_grad(), load_context:
            outputs_teacher = self.teacher_model(**teacher_inputs)
        teacher_logits = align_completion_logits(outputs_teacher.logits, n_comp)

        # --- vocabulary padding (student and teacher may differ) --------------
        stu_v = student_logits.shape[-1]
        tea_v = teacher_logits.shape[-1]
        if stu_v != tea_v:
            v = min(stu_v, tea_v)
            student_logits = student_logits[..., :v]
            teacher_logits = teacher_logits[..., :v]

        labels = build_completion_labels(inputs, n_comp, student_logits.device)

        loss = grasp_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits.detach(),
            labels=labels,
            beta=cfg.beta,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            token_clip=cfg.token_clip,
            reduction="batchmean",
        )

        self._grasp_loss_log(student_logits, teacher_logits, labels, loss, cfg)

        if return_outputs:
            return loss, outputs_student
        return loss

    def _grasp_loss_log(self, student_logits, teacher_logits, labels, loss, cfg):
        """Record the diagnostics that would reveal a silently-degenerate run."""
        try:
            mode = "train" if self.model.training else "eval"
            metrics = self._metrics[mode]
            with torch.no_grad():
                mask = labels != -100
                s_logp = torch.log_softmax(student_logits.float(), dim=-1)
                t_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
                fwd = (t_logp.exp() * (t_logp - s_logp)).sum(-1)[mask].mean()
                agree = (
                    (student_logits.argmax(-1) == teacher_logits.argmax(-1))[mask].float().mean()
                )
            metrics.setdefault(cfg.divergence_name, []).append(float(fwd))
            metrics.setdefault("grasp_argmax_agreement", []).append(float(agree))
            metrics.setdefault("grasp_supervised_tokens", []).append(float(mask.sum()))

            if self.accelerator.is_main_process and self.state.global_step % 20 == 0:
                _announce(
                    "[GRASP] step=%s loss=%.4f %s=%.4f argmax_agree=%.3f tokens=%d"
                    % (
                        self.state.global_step,
                        float(loss),
                        cfg.divergence_name,
                        float(fwd),
                        float(agree),
                        int(mask.sum()),
                    )
                )
        except Exception as exc:  # diagnostics must never break training
            _announce(f"[GRASP] metric logging failed: {exc!r}", level="warning")

    compute_loss._grasp_loss_patched = True
    GKDTrainer.compute_loss = compute_loss
    GKDTrainer._grasp_loss_log = _grasp_loss_log
    GKDTrainer._grasp_loss_patched = True
    GKDTrainer._grasp_loss_config = cfg
    _announce("[GRASP] compute_loss patched")
