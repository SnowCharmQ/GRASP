"""Judge cached teacher answers for per-aspect rubric coverage."""

from __future__ import annotations

import argparse
import json
import os
import sys

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{P}/src")
sys.path.insert(0, f"{P}/eval_src/src")

from rtv_score import RAW_DEFAULT, extract_answer  # noqa: E402
from logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default=f"{P}/outputs/rtv/teacher_gen.jsonl")
    ap.add_argument("--raw", default=RAW_DEFAULT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge-base-url", default="http://127.0.0.1:8100/v1,http://127.0.0.1:8101/v1")
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    gen = []
    with open(a.gen) as f:
        for line in f:
            gen.append(json.loads(line))
            if a.limit and len(gen) >= a.limit:
                break

    raw = {}
    with open(a.raw) as f:
        for line in f:
            r = json.loads(line)
            raw[r["id"]] = r

    miss = [g for g in gen if g["id"] not in raw]
    if miss:
        raise SystemExit(f"{len(miss)} ids absent from --raw (first {miss[0]['id']})")

    questions, details, responses, aspects_list, ids = [], [], [], [], []
    for g in gen:
        rr = raw[g["id"]]
        ids.append(g["id"])
        questions.append(rr["question"])
        details.append(rr.get("details") or "")
        # Judge the prose, not the JSON envelope -- same extraction the
        # evaluation path applies, so scores stay comparable.
        responses.append(extract_answer(g["teacher_output"]))
        aspects_list.append(rr.get("aspects") or [])

    n_asp = sum(len(x) for x in aspects_list)
    logger.info(f"  samples {len(ids)}, aspects {n_asp}")

    from judge import RubricJudge

    j = RubricJudge(base_url=a.judge_base_url, model=a.judge_model, concurrency=a.concurrency)
    scores = j.score_batch(questions, details, responses, aspects_list)
    # Never persist scores from a judge that was quietly failing: a dead server
    # returns 0.0, which is indistinguishable from "covered nothing".
    j.assert_healthy()

    with open(a.out, "w") as g:
        for i, s, asp, resp in zip(ids, scores, aspects_list, responses):
            g.write(
                json.dumps(
                    {
                        "id": i,
                        "rtv_score": s,
                        "n_aspects": len(asp),
                        "teacher_chars": len(resp),
                        "teacher_empty": not resp,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    v = sorted(x for x in scores if x is not None)
    logger.info(f"  wrote {len(scores)} -> {a.out}")
    if v:

        def q(f):
            return v[min(len(v) - 1, int(len(v) * f))]

        logger.info(
            f"  s(x) p10={q(0.10):.3f} p25={q(0.25):.3f} p50={q(0.50):.3f} "
            f"p75={q(0.75):.3f} p90={q(0.90):.3f}"
        )
        logger.info(
            f"  mean={sum(v) / len(v):.3f}  #(=0) {sum(1 for x in v if x == 0)}  "
            f"#(=1) {sum(1 for x in v if x == 1)}"
        )
        for t in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
            k = sum(1 for x in v if x >= t)
            logger.info(f"    tau={t:.1f} keep {k:6d}/{len(v)} ({100 * k / len(v):5.1f}%)")


if __name__ == "__main__":
    main()
