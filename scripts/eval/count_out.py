#!/usr/bin/env python3
"""Print "<total> <empty>" for a generate_responses output file.

The output file is the LaMP-QA response format, ``{qid: [{"output": ...}]}``.
The eval drivers use the first number to tell whether generation is complete
and the second to flag runs with too many blank answers.
"""
import json
import sys


def main() -> None:
    path = sys.argv[1]
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("0 0")
        return

    total = len(data)
    empty = 0
    for v in data.values():
        out = ""
        if isinstance(v, list) and v:
            out = (v[0] or {}).get("output", "") or ""
        elif isinstance(v, dict):
            out = v.get("output", "") or ""
        if not str(out).strip():
            empty += 1
    print(f"{total} {empty}")


if __name__ == "__main__":
    main()
