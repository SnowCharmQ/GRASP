"""Build JSONL student/teacher message pairs for swift.

Validate prompt invariants and record length/leakage drops in .stats.json.
Gemma folds the system message into the user turn."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts import (  # noqa: E402
    VALID_PI_MODES,
    PILeakInStudentContext,
    assert_template_invariant,
    build_pi_block,
    build_student_messages,
    build_teacher_messages,
)
from logging_utils import get_logger

logger = get_logger(__name__)


def _load_tokenizer(model_path):
    """Load a tokenizer for length accounting; ``None`` disables it."""
    if not model_path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _count_tokens(tokenizer, messages):
    """Count chat-template tokens, including the generation prompt suffix."""
    if tokenizer is None:
        text = "\n".join(m["content"] for m in messages if m.get("content"))
        return int(len(text.split()) * 1.3)
    try:
        ids = tokenizer.apply_chat_template(
            [m for m in messages if m.get("content")],
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(ids)
    except Exception:
        text = "\n".join(m["content"] for m in messages if m.get("content"))
        return len(tokenizer(text).input_ids)


def _fold_system_into_user(msgs):
    """Fold a leading system turn into the first user turn."""
    if not msgs or msgs[0]["role"] != "system":
        return msgs
    if len(msgs) < 2 or msgs[1]["role"] != "user":
        raise ValueError(f"unexpected role order: {[m['role'] for m in msgs]}")
    return [{"role": "user", "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}] + msgs[
        2:
    ]


def build(args):
    tokenizer = _load_tokenizer(args.tokenizer)
    if tokenizer is None:
        logger.warning(
            "no --tokenizer given: length filtering is DISABLED and only "
            "approximate lengths are reported. Pass the student model path."
        )

    stats = Counter()
    len_hist = {"student": [], "teacher": []}
    kept = []
    leaked_ids = []

    with open(args.input, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            stats["seen"] += 1
            row = json.loads(line)

            aspects = row.get("aspects") or []
            if args.pi_mode in {"aspects", "all"} and not aspects:
                stats["drop_no_aspects"] += 1
                continue

            student = build_student_messages(row, num_contexts=args.num_contexts)
            teacher = build_teacher_messages(
                row,
                num_contexts=args.teacher_num_contexts or args.num_contexts,
                pi_mode=args.pi_mode,
            )

            # Profile-width ablations use a weaker question-preservation check.
            same_width = (
                args.teacher_num_contexts is None or args.teacher_num_contexts == args.num_contexts
            )
            if same_width:
                try:
                    assert_template_invariant(
                        row, num_contexts=args.num_contexts, pi_mode=args.pi_mode
                    )
                except PILeakInStudentContext as exc:
                    # Drop and record rows whose profile already exposes the privileged
                    # context.
                    stats["drop_pi_leak"] += 1
                    leaked_ids.append(row.get("id"))
                    if stats["drop_pi_leak"] <= 5:
                        logger.warning(f"[drop:pi_leak] {exc}")
                    continue
            else:
                assert row["question"] in teacher[1]["content"]
                assert row["question"] in student[1]["content"]

            n_stu = _count_tokens(tokenizer, student)
            n_tea = _count_tokens(tokenizer, teacher)
            len_hist["student"].append(n_stu)
            len_hist["teacher"].append(n_tea)

            if tokenizer is not None:
                budget = args.max_prompt_tokens
                if n_stu > budget:
                    stats["drop_student_too_long"] += 1
                    continue
                if n_tea > budget:
                    stats["drop_teacher_too_long"] += 1
                    continue

            pi_block = build_pi_block(row.get("details"), aspects, args.pi_mode)
            if not pi_block and args.pi_mode != "none":
                stats["pi_empty_fellback_to_symmetric"] += 1

            # Gemma-2 cannot take a system role; fold it into the user turn.
            if args.family == "gemma":
                student = _fold_system_into_user(student)
                teacher = _fold_system_into_user(teacher)

            kept.append(
                {
                    # trailing empty assistant turn: swift's template encoder
                    # replaces it with the sampled completion.
                    "messages": student + [{"role": "assistant", "content": ""}],
                    "teacher_messages": teacher + [{"role": "assistant", "content": ""}],
                    "meta": {
                        "id": row.get("id"),
                        "config": row.get("config"),
                        "category": row.get("category"),
                        "n_aspects": len(aspects),
                        "n_profile": len(row.get("profile") or []),
                        "pi_chars": len(pi_block),
                        "student_prompt_tokens": n_stu,
                        "teacher_prompt_tokens": n_tea,
                    },
                }
            )
            stats["kept"] += 1

            if args.limit and stats["kept"] >= args.limit:
                break

    if args.shuffle:
        import random

        random.Random(args.seed).shuffle(kept)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for rec in kept:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _pct(vals, q):
        if not vals:
            return 0
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(len(vals) * q))]

    report = {
        "input": args.input,
        "output": args.output,
        "pi_mode": args.pi_mode,
        "num_contexts": args.num_contexts,
        "teacher_num_contexts": args.teacher_num_contexts,
        "max_prompt_tokens": args.max_prompt_tokens,
        "tokenizer": args.tokenizer,
        "counts": dict(stats),
        # Record rows whose retrieved profile exposes privileged narrative.
        "pi_leak_ids": leaked_ids,
        "student_prompt_tokens": {
            "p50": _pct(len_hist["student"], 0.50),
            "p95": _pct(len_hist["student"], 0.95),
            "max": max(len_hist["student"]) if len_hist["student"] else 0,
        },
        "teacher_prompt_tokens": {
            "p50": _pct(len_hist["teacher"], 0.50),
            "p95": _pct(len_hist["teacher"], 0.95),
            "max": max(len_hist["teacher"]) if len_hist["teacher"] else 0,
        },
    }
    with open(args.output + ".stats.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    logger.info(json.dumps(report, indent=2))

    # A dropped fraction above a few percent means the budget is misconfigured,
    # not that the data is bad. Fail loudly rather than train on a biased subset.
    dropped = (
        stats["drop_student_too_long"]
        + stats["drop_teacher_too_long"]
        + stats["drop_no_aspects"]
        + stats["drop_pi_leak"]
    )
    if stats["seen"] and dropped / stats["seen"] > args.max_drop_frac:
        raise SystemExit(
            f"ERROR: dropped {dropped}/{stats['seen']} "
            f"({dropped / stats['seen']:.1%}) > --max-drop-frac "
            f"{args.max_drop_frac:.1%}. Raise --max-prompt-tokens or lower "
            f"--num-contexts / --teacher-num-contexts."
        )
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        required=True,
        help="e.g. data/train.jsonl",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--family",
        default="qwen",
        choices=("qwen", "gemma"),
        help="target model family. 'gemma' folds the system turn into the user "
        "turn, since Gemma-2's chat template has no system role.",
    )
    ap.add_argument(
        "--pi-mode",
        default="all",
        choices=VALID_PI_MODES,
        help="which privileged channels the teacher sees (main ablation axis)",
    )
    ap.add_argument(
        "--num-contexts",
        type=int,
        default=10,
        help="profile entries for the STUDENT (official LaMP-QA uses k=10)",
    )
    ap.add_argument(
        "--teacher-num-contexts",
        type=int,
        default=None,
        help="profile entries for the TEACHER; set >num-contexts for the S3 "
        "'retrieval privilege' ablation. Default: same as the student.",
    )
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="student model path, used for length filtering. Strongly recommended.",
    )
    ap.add_argument("--max-prompt-tokens", type=int, default=6144)
    ap.add_argument("--max-drop-frac", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
