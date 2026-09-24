"""Apply GRASP context and loss patches before starting the swift RLHF CLI."""

from __future__ import annotations

import os
import sys
from logging_utils import get_logger

logger = get_logger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from grasp_patch import apply_grasp_patch

    patched = apply_grasp_patch()

    # Apply the context patch before the loss patch.
    from grasp_loss import apply_grasp_loss_patch

    apply_grasp_loss_patch()
    # Use the project logger so framework root-logger changes do not hide activation.
    banner = (
        f"[GRASP] patch {'APPLIED' if patched else 'NOT applied'} "
        f"(LAMPQA_GRASP={os.environ.get('LAMPQA_GRASP', '<unset>')}, "
        f"rank={os.environ.get('RANK', '?')})"
    )
    get_logger(__name__).info("%s", banner)

    # Fail fast on configurations the patch cannot honour, before any model loads.
    if patched:
        argv = " ".join(sys.argv[1:])
        for flag, why in (
            ("--padding_free true", "padding_free flattens the batch"),
            ("--sequence_parallel_size", "sequence parallelism shards the seq dim"),
            ("--seq_kd true", "seq_kd makes the teacher generate, not the student"),
        ):
            if flag in argv:
                if flag == "--sequence_parallel_size":
                    # only a value > 1 is a problem
                    try:
                        parts = argv.split(flag)[1].split()
                        if parts and int(parts[0]) <= 1:
                            continue
                    except (IndexError, ValueError):
                        pass
                raise SystemExit(
                    f"[GRASP] refusing to run: {flag} is incompatible ({why}). "
                    f"See grasp_patch.assert_supported_config."
                )

    # The importable entry point lives in swift.llm; swift/cli/rlhf.py is only a
    # `if __name__ == '__main__'` shim and exports no symbol.
    from swift.llm import rlhf_main

    rlhf_main()


if __name__ == "__main__":
    main()
