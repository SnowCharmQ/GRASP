"""Give the teacher privileged context while scoring the student completion.

Teacher batches are left-padded and completion tokens must stay aligned.
Padding-free and sequence-parallel training are unsupported."""

from __future__ import annotations

import os

import torch
from logging_utils import get_logger

logger = get_logger(__name__)


_GRASP_ENV = "LAMPQA_GRASP"
_TEACHER_PROMPT_KEY = "_grasp_teacher_prompt_ids"


def _announce(msg, level="info"):
    """Emit a message that survives the engine's logging reconfiguration."""
    getattr(get_logger(__name__), level)("%s", msg)


# Pure tensor helpers (unit-testable without a model)


def extract_completion_spans(input_ids, labels, pad_token_id):
    """Return, per row, the completion token ids implied by ``labels != -100``."""
    assert input_ids.shape == labels.shape, (
        f"input_ids {tuple(input_ids.shape)} != labels {tuple(labels.shape)}"
    )
    spans = []
    for row_ids, row_labels in zip(input_ids, labels):
        mask = row_labels != -100
        spans.append(row_ids[mask])
    return spans


def build_teacher_batch(
    teacher_prompt_ids,
    completion_spans,
    pad_token_id,
    device=None,
    dtype=torch.long,
):
    """Concatenate ``[teacher_prompt | completion]`` per row and pad on the LEFT."""
    assert len(teacher_prompt_ids) == len(completion_spans), (
        f"batch mismatch: {len(teacher_prompt_ids)} teacher prompts vs "
        f"{len(completion_spans)} completions"
    )

    rows, comp_lens = [], []
    for prompt, comp in zip(teacher_prompt_ids, completion_spans):
        prompt = torch.as_tensor(prompt, dtype=dtype)
        comp = torch.as_tensor(comp, dtype=dtype)
        rows.append(torch.cat([prompt, comp], dim=0))
        comp_lens.append(int(comp.numel()))

    max_len = max(int(r.numel()) for r in rows)
    bsz = len(rows)

    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=dtype)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=dtype)

    for i, (row, clen) in enumerate(zip(rows, comp_lens)):
        n = int(row.numel())
        start = max_len - n  # left padding
        input_ids[i, start:] = row
        attention_mask[i, start:] = 1
        if clen:
            labels[i, max_len - clen :] = row[-clen:]

    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

    return input_ids, attention_mask, labels, comp_lens


def assert_completion_alignment(
    student_input_ids,
    student_labels,
    teacher_input_ids,
    teacher_labels,
):
    """Assert both sides carry the *same* completion tokens in the same order."""
    s_spans = extract_completion_spans(student_input_ids, student_labels, 0)
    t_spans = extract_completion_spans(teacher_input_ids, teacher_labels, 0)

    assert len(s_spans) == len(t_spans), (
        f"batch size mismatch: student {len(s_spans)} vs teacher {len(t_spans)}"
    )
    for i, (s, t) in enumerate(zip(s_spans, t_spans)):
        assert s.numel() == t.numel(), (
            f"row {i}: completion length differs -- student {s.numel()} vs "
            f"teacher {t.numel()}. The teacher prompt/completion concat is wrong."
        )
        if not torch.equal(s.cpu(), t.cpu()):
            first = int((s.cpu() != t.cpu()).nonzero()[0].item())
            raise AssertionError(
                f"row {i}: completion token ids differ at offset {first} "
                f"(student {int(s[first])} vs teacher {int(t[first])}). The "
                f"teacher is scoring a DIFFERENT sequence than the student "
                f"sampled; the reverse KL would be meaningless."
            )
    return True


# The patch


def _grasp_enabled():
    return os.environ.get(_GRASP_ENV, "0").lower() in ("1", "true", "yes")


def patch_engine_logps_fallback():
    """Install text-only log-probability fallback on RolloutTrainerMixin.
    GKDTrainer calls it through super(), so patching GKDTrainer is insufficient."""
    from swift.trainers.rlhf_trainer.gkd_trainer import GKDTrainer

    # Patch the mixin: super() lookup skips GKDTrainer itself.
    mro = GKDTrainer.__mro__
    target = None
    for cls in mro[1:]:
        if cls.__module__.startswith("swift."):
            target = cls
            break
    if target is None:
        raise RuntimeError(
            f"no swift-owned class after GKDTrainer in the MRO: {[c.__name__ for c in mro]}"
        )

    if "_get_per_token_logps" in target.__dict__:
        return False

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        """Match the engine local-forward slicing, including causal shift and temperature."""
        from trl.trainer.utils import selective_log_softmax

        if isinstance(logits_to_keep, torch.Tensor):
            logits_to_keep = int(logits_to_keep.max().item())
        logits_to_keep = int(logits_to_keep)

        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "logits_to_keep" in getattr(self, "model_kwarg_keys", ()):
            kwargs["logits_to_keep"] = logits_to_keep + 1

        logits = model(**kwargs).logits
        logits = logits[:, -(logits_to_keep + 1) : -1, :] / getattr(self, "temperature", 1.0)
        target_ids = input_ids[:, -logits_to_keep:]
        return selective_log_softmax(logits, target_ids)

    _get_per_token_logps._grasp_engine_fallback = True
    target._get_per_token_logps = _get_per_token_logps
    _announce(
        f"[GRASP] installed _get_per_token_logps fallback on {target.__name__} "
        f"(engine expects it on super(), but no trl version provides it)"
    )
    return True


def patch_preprocessor_keep_teacher_messages():
    """Keep teacher_messages in the swift preprocessor column whitelist."""
    from swift.llm.dataset.preprocessor.core import RowPreprocessor

    for key in ("teacher_messages", "meta"):
        if key not in RowPreprocessor.standard_keys:
            RowPreprocessor.standard_keys = RowPreprocessor.standard_keys + [key]
    _announce(
        "[GRASP] whitelisted 'teacher_messages' + 'meta' in "
        "RowPreprocessor.standard_keys (swift would otherwise select_columns "
        "them away)"
    )
    return True


def _grasp_install_nocompile_guard():
    """Disable generation compilation for Gemma hybrid caches to avoid
    DeepSpeed/Dynamo incompatibility."""
    from swift.trainers.rlhf_trainer.gkd_trainer import GKDTrainer

    if getattr(GKDTrainer, "_grasp_nocompile_patched", False):
        return

    _orig = GKDTrainer.generate_on_policy_outputs

    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        if (
            generation_config is not None
            and getattr(generation_config, "disable_compile", None) is not True
        ):
            try:
                generation_config.disable_compile = True
            except Exception:
                pass
        return _orig(self, model, inputs, generation_config, pad_token_id)

    GKDTrainer.generate_on_policy_outputs = generate_on_policy_outputs
    GKDTrainer._grasp_nocompile_patched = True
    _announce("[GRASP] installed no-compile guard for generate() (gemma-2 hybrid cache)")


def apply_grasp_patch():
    """Monkey-patch ``GKDTrainer`` for privileged-teacher OPD."""
    # Needed by both arms, so it is applied before the LAMPQA_GRASP check.
    patch_engine_logps_fallback()
    # Also needed by both arms: gemma-2 would otherwise crash in dynamo
    # on the OPD path, where the GRASP wrapper below is never installed.
    _grasp_install_nocompile_guard()

    if not _grasp_enabled():
        _announce(f"[GRASP] {_GRASP_ENV} not set -- running vanilla OPD (S1)")
        return False

    # Must run before the dataset is preprocessed (i.e. before the trainer is
    # constructed), which is why the launcher patches as its very first action.
    patch_preprocessor_keep_teacher_messages()

    from swift.trainers.rlhf_trainer.gkd_trainer import GKDTrainer

    if getattr(GKDTrainer, "_grasp_patched", False):
        _announce("[GRASP] already patched")
        return True

    orig_prepare_batch = GKDTrainer._prepare_batch_inputs
    orig_compute_loss = GKDTrainer.compute_loss
    orig_get_logps = GKDTrainer._get_per_token_logps_and_entropies
    orig_generate = GKDTrainer.generate_on_policy_outputs

    def _encode_teacher_prompts(self, inputs):
        """Tokenize each row's ``teacher_messages`` into prompt-only ids."""
        template = self.template
        out = []
        with self._template_context(template, mode="pt"):
            for data in inputs:
                tmsgs = data.get("teacher_messages")
                if not tmsgs:
                    raise KeyError(
                        "GRASP is enabled but a row has no 'teacher_messages'. "
                        "Rebuild the dataset with src/build_data.py."
                    )
                msgs = [dict(m) for m in tmsgs]
                if msgs and msgs[-1].get("role") == "assistant":
                    msgs[-1]["content"] = None
                encoded = template.encode({"messages": msgs}, return_length=True)
                ids = encoded["input_ids"]
                if isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                out.append(ids)
        return out

    def _prepare_batch_inputs(self, inputs, encode_prompt_only=False):
        # Encode the teacher prompts *before* the student encoding mutates
        # `inputs` (it sets the assistant content to None in place).
        if not getattr(self, "_grasp_seen_prepare", False):
            # Log retained columns once to diagnose dropped teacher metadata.
            self._grasp_seen_prepare = True
            keys = (
                sorted(inputs[0].keys())
                if isinstance(inputs, list) and inputs and isinstance(inputs[0], dict)
                else "n/a"
            )
            _announce(f"[GRASP] first batch columns: {keys}")

        teacher_prompts = None
        if isinstance(inputs, list) and inputs and isinstance(inputs[0], dict):
            if any("teacher_messages" in d for d in inputs):
                teacher_prompts = _encode_teacher_prompts(self, inputs)
            elif not getattr(self, "_grasp_warned_missing", False):
                # Report missing teacher context instead of silently using symmetric
                # prompts.
                self._grasp_warned_missing = True
                keys = sorted(inputs[0].keys()) if isinstance(inputs[0], dict) else None
                _announce(
                    "[GRASP][FATAL] LAMPQA_GRASP=1 but no row in this batch carries "
                    "'teacher_messages'. swift's preprocessing has dropped the "
                    f"column. Row keys seen: {keys}. Training would silently "
                    "become self-distillation (reverse_kl == 0).",
                    level="error",
                )
                raise KeyError(
                    "GRASP enabled but 'teacher_messages' is absent from the batch; "
                    f"row keys = {keys}. Register the column with swift (see "
                    "--columns / a custom preprocessor) or disable LAMPQA_GRASP."
                )

        batch = orig_prepare_batch(self, inputs, encode_prompt_only=encode_prompt_only)

        if teacher_prompts is not None:
            batch[_TEACHER_PROMPT_KEY] = teacher_prompts
        return batch

    def _grasp_build_teacher_inputs(self, inputs, teacher_prompts=None):
        """Build the teacher's forward inputs with its own (longer) prompt."""
        if teacher_prompts is None:
            teacher_prompts = inputs.get(_TEACHER_PROMPT_KEY)
        if teacher_prompts is None:
            raise KeyError(
                f"GRASP enabled but '{_TEACHER_PROMPT_KEY}' missing from the batch; "
                "the _prepare_batch_inputs patch did not run."
            )

        student_ids = inputs["input_ids"]
        student_labels = inputs.get("labels")
        pad_id = self.processing_class.pad_token_id
        if pad_id is None:
            pad_id = getattr(self.processing_class, "eos_token_id", 0) or 0

        # Prefer logits_to_keep because labels may already be completion-only.
        n_keep = inputs.get("logits_to_keep")
        if isinstance(n_keep, torch.Tensor):
            n_keep = int(n_keep.max().item())

        if n_keep:
            n_keep = int(n_keep)
            comp_spans = [row[-n_keep:].tolist() for row in student_ids]
        elif student_labels is not None and student_labels.shape == student_ids.shape:
            comp_spans = extract_completion_spans(student_ids, student_labels, pad_id)
        else:
            raise ValueError(
                "cannot locate the completion span: no usable 'logits_to_keep' "
                f"and labels shape {None if student_labels is None else tuple(student_labels.shape)} "
                f"does not match input_ids {tuple(student_ids.shape)}"
            )

        t_ids, t_mask, t_labels, comp_lens = build_teacher_batch(
            teacher_prompts,
            comp_spans,
            pad_token_id=pad_id,
            device=student_ids.device,
            dtype=student_ids.dtype,
        )

        # Verify that both scoring windows contain the same completion tokens.
        for i, span in enumerate(comp_spans):
            k = len(span)
            teacher_tail = t_ids[i, -k:].tolist()
            if teacher_tail != list(span):
                raise AssertionError(
                    f"row {i}: teacher's last {k} tokens do not match the "
                    f"student's completion. This is the misalignment that makes "
                    f"the KL compare p_student(y_t) with p_teacher(y_t+1) and "
                    f"still looks like a healthy training curve.\n"
                    f"  student: {list(span)[:8]} ...\n"
                    f"  teacher: {teacher_tail[:8]} ..."
                )
        if student_labels is not None and student_labels.shape == student_ids.shape:
            # Full-width labels available: run the stricter original check too.
            assert_completion_alignment(student_ids, student_labels, t_ids, t_labels)

        teacher_inputs = {
            "input_ids": t_ids,
            "attention_mask": t_mask,
            "labels": t_labels,
            # max over rows: left padding keeps every completion right-aligned,
            # so this slice covers all of them.
            "logits_to_keep": max(comp_lens),
        }
        if "position_ids" in inputs:
            teacher_inputs["position_ids"] = (t_mask.long().cumsum(-1) - 1).clamp_(min=0)
        return teacher_inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Keep privileged prompt IDs available during the engine loss call.
        The log-probability wrapper substitutes teacher inputs before scoring."""
        self._grasp_active_prompt_ids = inputs.get(_TEACHER_PROMPT_KEY)
        inputs = dict(inputs)
        inputs.pop(_TEACHER_PROMPT_KEY, None)
        try:
            return orig_compute_loss(
                self,
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        finally:
            self._grasp_active_prompt_ids = None

    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        """Hide teacher metadata from model.generate(), then restore it for loss scoring."""
        stashed = inputs.pop(_TEACHER_PROMPT_KEY, None)
        # Disable compilation on the live generation config; swift rebuilds it.
        if (
            generation_config is not None
            and getattr(generation_config, "disable_compile", None) is not True
        ):
            try:
                generation_config.disable_compile = True
            except Exception:
                pass
        try:
            return orig_generate(self, model, inputs, generation_config, pad_token_id)
        finally:
            if stashed is not None:
                inputs[_TEACHER_PROMPT_KEY] = stashed

    def _get_per_token_logps_and_entropies(self, model, inputs, compute_entropy=False, **kwargs):
        """Route teacher scoring through privileged inputs; leave student scoring unchanged."""
        is_teacher = (
            getattr(self, "teacher_model", None) is not None and model is self.teacher_model
        )
        prompt_ids = getattr(self, "_grasp_active_prompt_ids", None)

        if is_teacher and prompt_ids is not None:
            teacher_inputs = _grasp_build_teacher_inputs(self, inputs, prompt_ids)
            if self.accelerator.is_main_process and self.state.global_step % 20 == 0:
                _announce(
                    "[GRASP] teacher forward on privileged prompt: "
                    "student_len=%d teacher_len=%d logits_to_keep=%d"
                    % (
                        inputs["input_ids"].shape[1],
                        teacher_inputs["input_ids"].shape[1],
                        teacher_inputs["logits_to_keep"],
                    )
                )
            inputs = teacher_inputs

        return orig_get_logps(self, model, inputs, compute_entropy=compute_entropy, **kwargs)

    # Tag replacement methods so later overrides can be detected.
    _prepare_batch_inputs._grasp_patched = True
    _prepare_batch_inputs.__wrapped__ = orig_prepare_batch
    compute_loss._grasp_patched = True
    compute_loss.__wrapped__ = orig_compute_loss
    _get_per_token_logps_and_entropies._grasp_patched = True
    _get_per_token_logps_and_entropies.__wrapped__ = orig_get_logps
    generate_on_policy_outputs._grasp_patched = True
    generate_on_policy_outputs.__wrapped__ = orig_generate

    GKDTrainer._prepare_batch_inputs = _prepare_batch_inputs
    GKDTrainer._grasp_build_teacher_inputs = _grasp_build_teacher_inputs
    GKDTrainer.compute_loss = compute_loss
    GKDTrainer._get_per_token_logps_and_entropies = _get_per_token_logps_and_entropies
    GKDTrainer.generate_on_policy_outputs = generate_on_policy_outputs
    GKDTrainer._grasp_patched = True

    _announce("[GRASP] patched GKDTrainer for privileged-teacher OPD")
    return True


def assert_supported_config(args):
    """Reject configurations this patch cannot handle, loudly and early."""
    problems = []
    if getattr(args, "padding_free", False):
        problems.append(
            "padding_free=True flattens the batch; the teacher concat assumes "
            "a rectangular [bsz, seq] layout"
        )
    if getattr(args, "sequence_parallel_size", 1) and (
        getattr(args, "sequence_parallel_size", 1) > 1
    ):
        problems.append(
            "sequence_parallel_size>1 shards the sequence dimension across ranks; "
            "the teacher's longer prompt would shard differently from the student's"
        )
    if getattr(args, "seq_kd", False):
        problems.append(
            "seq_kd=True makes the TEACHER generate the trajectory, which defeats "
            "on-policy distillation (the student must sample its own)"
        )
    if problems:
        raise SystemExit("[GRASP] unsupported configuration:\n  - " + "\n  - ".join(problems))
    return True
