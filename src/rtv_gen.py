"""Cache teacher answers separately from judging so generations can be reused."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{P}/src")
sys.path.insert(0, f"{P}/eval_src/src")

from rtv_score import _generate, teacher_prompt  # noqa: E402
from logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{P}/data/processed/train_pi_aspects.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8500/v1")
    ap.add_argument("--model", default="policy")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()

    rows = []
    with open(a.data) as f:
        for line in f:
            rows.append(json.loads(line))
            if a.limit and len(rows) >= a.limit:
                break

    # Reuse cached generations when resuming.
    done = {}
    if os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["id"]] = r["teacher_output"]
                except Exception:
                    pass
        logger.info(f"  {len(done)} already done, skipping")

    todo = [r for r in rows if r["meta"]["id"] not in done]
    logger.info(f"  to generate: {len(todo)}/{len(rows)}")
    if not todo:
        logger.info("  nothing to do")
        return

    prompts = [teacher_prompt(r) for r in todo]
    outs = asyncio.run(
        _generate(
            prompts, a.base_url, a.model, a.temperature, a.top_p, a.max_tokens, a.concurrency, False
        )
    )

    with open(a.out, "a") as g:
        for r, o in zip(todo, outs):
            g.write(
                json.dumps({"id": r["meta"]["id"], "teacher_output": o or ""}, ensure_ascii=False)
                + "\n"
            )

    empty = sum(1 for o in outs if not (o or "").strip())
    logger.info(f"  wrote {len(outs)}, empty {empty}")


if __name__ == "__main__":
    main()
