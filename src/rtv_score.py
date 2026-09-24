"""Generate teacher answers, judge rubric coverage, and cache RTV scores."""

from __future__ import annotations

import argparse
import asyncio
import re
import json
import os
import sys
from logging_utils import get_logger

logger = get_logger(__name__)

# Load rubric text from raw training rows; processed rows only retain counts.
RAW_DEFAULT = "data/raw/train.jsonl"

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "eval_src", "src"))


def teacher_prompt(row: dict) -> list[dict]:
    """The teacher side of an OPSD pair, privileged context included."""
    return [m for m in row["teacher_messages"] if m["role"] != "assistant"]


def extract_answer(text: str) -> str:
    """Pull the answer out of the teacher's JSON envelope."""
    from generate_responses import _parse_personalized_answer

    if not (text or "").strip():
        return ""
    try:
        got = _parse_personalized_answer(text)
    except Exception:
        got = None
    if got and str(got).strip():
        return str(got).strip()
    if os.environ.get("LAMPQA_RTV_LENIENT") == "1":
        return _lenient_answer(text)
    return ""


_LEN_FIELD = re.compile(r'"personalized_answer"\s*:\s*"', re.S)
_LEN_ALT = re.compile(
    r'"(?:response|answer|personalised_answer|personalized_response|text)"'
    r'\s*:\s*"',
    re.S | re.I,
)
_LEN_FENCE = re.compile(r"^\s*```(?:json)?\s*", re.S)
_LEN_SIGNOFF = re.compile(
    r"\n\s*(?:Let me know if|I hope this helps|Feel free to|"
    r"Would you like me to|Hope that helps)",
    re.I,
)


def _lenient_tidy(s: str) -> str:
    """Trim the JSON tail the model failed to close, plus chat sign-offs."""
    s = (s or "").strip()
    ends = [i for i in (s.find('"\n}'), s.find('"}'), s.find("\n```")) if i > 0]
    if ends:
        s = s[: min(ends)]
    s = s.rstrip().rstrip('"').rstrip()
    s = s.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
    return _LEN_SIGNOFF.split(s, maxsplit=1)[0].strip()


def _lenient_answer(text: str) -> str:
    """Recover prose from a reply that broke the JSON contract."""
    m = _LEN_FIELD.search(text)
    if m:
        body = _lenient_tidy(text[m.end() :])
        if body:
            return body
    m = _LEN_ALT.search(text)
    if m:
        body = _lenient_tidy(text[m.end() :])
        if body:
            return body
    body = _LEN_FENCE.sub("", text)
    if body.lstrip().startswith("{") and '":' in body:
        return ""
    return _lenient_tidy(body)


async def _generate(
    prompts, base_url, model, temperature, top_p, max_tokens, concurrency, merge_system
):
    """Drive eval_src's single-request helper over the whole batch."""
    from openai import AsyncOpenAI

    from generate_responses import _one, merge_system_into_user

    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    sem = asyncio.Semaphore(concurrency)
    if merge_system:
        prompts = [merge_system_into_user(p) for p in prompts]

    tasks = [_one(client, sem, model, p, temperature, top_p, max_tokens) for p in prompts]
    done = 0
    out = [None] * len(tasks)

    async def track(i, coro):
        nonlocal done
        out[i] = await coro
        done += 1
        if done % 200 == 0 or done == len(tasks):
            logger.info(f"generated {done}/{len(tasks)}")

    await asyncio.gather(*(track(i, t) for i, t in enumerate(tasks)))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="OPSD jsonl with messages/teacher_messages/meta")
    p.add_argument(
        "--raw",
        default=RAW_DEFAULT,
        help="split jsonl carrying question/details/aspects; "
        "defaults to the ranked train split, the only file "
        "that has them for the train ids",
    )
    p.add_argument("--out", required=True, help="jsonl of per-sample scores")
    p.add_argument(
        "--gen-base-url", required=True, help="OpenAI-compatible endpoint serving the teacher"
    )
    p.add_argument("--gen-model", default="policy")
    p.add_argument(
        "--judge-base-url",
        required=True,
        help="endpoint for the validating judge; keep it distinct from the evaluation judge",
    )
    p.add_argument("--judge-model", default="judge")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--merge-system",
        action="store_true",
        help="fold system into the first user turn (Gemma-2 has no system role)",
    )
    args = p.parse_args()

    raw = {}
    with open(args.raw) as f:
        for line in f:
            r = json.loads(line)
            raw[r["id"]] = r

    rows = []
    with open(args.data) as f:
        for line in f:
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    logger.info(f"{len(rows)} training samples, {len(raw)} raw records")

    missing = [r for r in rows if r["meta"]["id"] not in raw]
    if missing:
        raise SystemExit(
            f"[rtv] {len(missing)} samples have no raw record (first: "
            f"{missing[0]['meta']['id']}). Wrong --raw split?"
        )

    # 1. Teacher generations, privileged context included.
    prompts = [teacher_prompt(r) for r in rows]
    logger.info(f"generating {len(prompts)} teacher answers")
    outs = asyncio.run(
        _generate(
            prompts,
            args.gen_base_url,
            args.gen_model,
            args.temperature,
            args.top_p,
            args.max_tokens,
            args.concurrency,
            args.merge_system,
        )
    )

    empty = sum(1 for o in outs if not (o or "").strip())
    if empty:
        logger.warning(
            f"{empty}/{len(outs)} teacher outputs are empty; they will score 0 and be filtered out"
        )
    unparsed = sum(1 for o in outs if (o or "").strip() and not extract_answer(o))
    if unparsed:
        logger.warning(
            f"{unparsed}/{len(outs)} teacher outputs did not "
            f"honour the JSON contract; they score 0 by design"
        )

    # 2. Per-aspect coverage from the validating judge.
    from judge import RubricJudge

    judge = RubricJudge(base_url=args.judge_base_url, model=args.judge_model)

    questions, details, responses, aspects_list = [], [], [], []
    for r, o in zip(rows, outs):
        rr = raw[r["meta"]["id"]]
        questions.append(rr["question"])
        details.append(rr.get("details") or rr.get("narrative") or "")
        responses.append(extract_answer(o))
        aspects_list.append(rr.get("aspects") or [])

    n_asp = sum(len(a) for a in aspects_list)
    logger.info(f"judging {n_asp} aspects over {len(rows)} samples")
    scores = judge.score_batch(questions, details, responses, aspects_list)
    # Refuse to persist scores from a judge that was silently failing.
    judge.assert_healthy()

    with open(args.out, "w") as g:
        for r, o, s, asp in zip(rows, outs, scores, aspects_list):
            g.write(
                json.dumps(
                    {
                        "id": r["meta"]["id"],
                        "rtv_score": s,
                        "n_aspects": len(asp),
                        "teacher_chars": len(extract_answer(o)),
                        "teacher_raw_chars": len(o or ""),
                        "teacher_empty": not extract_answer(o),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    vals = sorted(s for s in scores if s is not None)
    if vals:

        def q(f):
            return vals[min(len(vals) - 1, int(len(vals) * f))]

        logger.info(f"wrote {len(scores)} -> {args.out}")
        logger.info(
            f"score p10={q(0.10):.3f} p25={q(0.25):.3f} "
            f"p50={q(0.50):.3f} p75={q(0.75):.3f} p90={q(0.90):.3f}"
        )
        for tau in (0.0, 0.4, 0.6, 0.8):
            kept = sum(1 for v in vals if v >= tau)
            logger.info(
                f"tau={tau:.1f} -> keep {kept:5d}/{len(vals)} ({100 * kept / len(vals):5.1f}%)"
            )


if __name__ == "__main__":
    main()
