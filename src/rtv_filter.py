"""Filter by RTV threshold and report retention and discarded-sample statistics."""

from __future__ import annotations

import argparse
import json
import statistics as st
from logging_utils import get_logger

logger = get_logger(__name__)


def load_scores(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def pct(v, f):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * f))] if v else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="OPSD jsonl to filter")
    p.add_argument("--scores", required=True, help="output of rtv_score.py")
    p.add_argument("--raw", default=None, help="split jsonl, enables profile-size analysis")
    p.add_argument("--out", default=None, help="filtered jsonl; omit to only log the sweep")
    p.add_argument("--tau", type=float, default=0.6)
    p.add_argument(
        "--sweep", default="0.0,0.4,0.6,0.8", help="thresholds to report before filtering"
    )
    args = p.parse_args()

    scores = load_scores(args.scores)
    rows = []
    with open(args.data) as f:
        for line in f:
            rows.append(json.loads(line))

    raw = {}
    if args.raw:
        with open(args.raw) as f:
            for line in f:
                r = json.loads(line)
                raw[r["id"]] = r

    have = [r for r in rows if r["meta"]["id"] in scores]
    if len(have) != len(rows):
        logger.warning(
            f"{len(rows) - len(have)} samples have no score; "
            f"they are treated as unscored and dropped"
        )

    vals = [scores[r["meta"]["id"]]["rtv_score"] for r in have]
    vals = [v for v in vals if v is not None]

    logger.info(f"{len(rows)} samples, {len(vals)} scored")
    logger.info(
        f"score p10={pct(vals, 0.1):.3f} p50={pct(vals, 0.5):.3f} "
        f"p90={pct(vals, 0.9):.3f}  mean={st.mean(vals):.3f}"
    )

    logger.info("\n=== tau sweep ===")
    logger.info(f"  {'tau':>5} {'kept':>7} {'ratio':>7}")
    for t in [float(x) for x in args.sweep.split(",")]:
        k = sum(1 for v in vals if v >= t)
        logger.info(f"  {t:5.2f} {k:7d} {100 * k / len(vals):6.1f}%")

    # Characterise the split at the chosen threshold.
    keep, drop = [], []
    for r in have:
        s = scores[r["meta"]["id"]]
        if s["rtv_score"] is not None and s["rtv_score"] >= args.tau:
            keep.append((r, s))
        else:
            drop.append((r, s))

    logger.info(
        f"\n=== tau={args.tau}: keep {len(keep)}, drop {len(drop)} "
        f"({100 * len(drop) / max(1, len(have)):.1f}%) ==="
    )

    def describe(name, bucket):
        if not bucket:
            logger.info(f"  {name}: (empty)")
            return
        nasp = [s["n_aspects"] for _, s in bucket]
        tchars = [s["teacher_chars"] for _, s in bucket]
        empty = sum(1 for _, s in bucket if s["teacher_empty"])
        line = (
            f"  {name:6s} n={len(bucket):6d}  "
            f"n_aspects p50={pct(nasp, 0.5):.0f} mean={st.mean(nasp):.2f}  "
            f"teacher_chars p50={pct(tchars, 0.5):.0f}  empty={empty}"
        )
        if raw:
            prof = [
                len(raw[r["meta"]["id"]].get("profile") or [])
                for r, _ in bucket
                if r["meta"]["id"] in raw
            ]
            if prof:
                line += f"  profile p50={pct(prof, 0.5):.0f}"
        logger.info(line)

    describe("keep", keep)
    describe("drop", drop)

    # Is the drop rate monotone in aspect count? That is the "hard case"
    # hypothesis stated when RTV was specified.
    logger.info("\n=== drop rate by aspect count ===")
    buckets: dict[int, list[float]] = {}
    for r, s in have and [(r, scores[r["meta"]["id"]]) for r in have]:
        if s["rtv_score"] is None:
            continue
        buckets.setdefault(min(s["n_aspects"], 6), []).append(s["rtv_score"])
    logger.info(f"  {'n_aspects':>10} {'n':>7} {'mean_s':>8} {'drop%':>7}")
    for k in sorted(buckets):
        v = buckets[k]
        d = sum(1 for x in v if x < args.tau)
        label = f"{k}" if k < 6 else "6+"
        logger.info(f"  {label:>10} {len(v):7d} {st.mean(v):8.3f} {100 * d / len(v):6.1f}%")

    if args.out:
        with open(args.out, "w") as g:
            for r, _ in keep:
                g.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"\nwrote {len(keep)} samples -> {args.out}")
        meta = args.out + ".stats.json"
        with open(meta, "w") as g:
            json.dump(
                {
                    "source": args.data,
                    "scores": args.scores,
                    "tau": args.tau,
                    "n_in": len(rows),
                    "n_scored": len(vals),
                    "n_kept": len(keep),
                    "n_dropped": len(drop),
                    "keep_ratio": len(keep) / max(1, len(have)),
                    "score_mean": st.mean(vals),
                    "score_p50": pct(vals, 0.5),
                },
                g,
                indent=2,
            )
        logger.info(f"wrote provenance -> {meta}")


if __name__ == "__main__":
    main()
