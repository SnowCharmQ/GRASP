import math

import pytest

from model.monkey_patch_timestamp_format import (
    Qwen3VLProcessor,
    apply_timestamp_format_patch,
    build_timestamp_formatter,
    parse_ts_prompt,
)


def _reset_patch_state():
    # Reset monkey patch flags to allow repeatable tests
    if hasattr(Qwen3VLProcessor, "_timestamp_patch_installed"):
        Qwen3VLProcessor._timestamp_patch_installed = False
    if hasattr(Qwen3VLProcessor, "_timestamp_formatter"):
        delattr(Qwen3VLProcessor, "_timestamp_formatter")
    if hasattr(Qwen3VLProcessor, "_timestamp_wrap_brackets"):
        delattr(Qwen3VLProcessor, "_timestamp_wrap_brackets")
    if hasattr(Qwen3VLProcessor, "_timestamp_append_colon"):
        delattr(Qwen3VLProcessor, "_timestamp_append_colon")


@pytest.fixture(autouse=True)
def _clean_patch():
    _reset_patch_state()
    yield
    _reset_patch_state()


def test_parse_ts_prompt_recognizes_keys():
    opts = parse_ts_prompt("seconds;unit=0;brackets=1;append_colon=1;decimals=3;trailing_colon=0")
    assert opts["style"] == "seconds"
    assert opts["include_unit"] is False
    assert opts["wrap_with_brackets"] is True
    assert opts["append_colon"] is True
    assert opts["decimal_places"] == 3
    assert opts["trailing_colon"] is False


def test_build_timestamp_formatter_seconds_precision_and_unit():
    fmt = build_timestamp_formatter(style="seconds", include_unit=True, decimal_places=2)
    assert fmt(0.234) == "0.23s"
    fmt_no_unit = build_timestamp_formatter(style="seconds", include_unit=False, decimal_places=1)
    assert fmt_no_unit(0.234) == "0.2"


def test_apply_patch_uses_ts_prompt_over_style():
    apply_timestamp_format_patch(ts_prompt="hms", format_style="seconds")
    fmt = Qwen3VLProcessor._timestamp_formatter
    wrapped = getattr(Qwen3VLProcessor, "_timestamp_wrap_brackets", None)
    appended = getattr(Qwen3VLProcessor, "_timestamp_append_colon", None)

    out = fmt(5)
    rep = f"<{out}>" if wrapped else out
    rep = f"{rep}:" if appended else rep
    assert rep == "<00:00:05>"


def test_apply_patch_brackets_and_append_colon_seconds():
    apply_timestamp_format_patch(ts_prompt="seconds;unit=1;brackets=1;append_colon=1;decimals=1")
    fmt = Qwen3VLProcessor._timestamp_formatter
    wrapped = getattr(Qwen3VLProcessor, "_timestamp_wrap_brackets", None)
    appended = getattr(Qwen3VLProcessor, "_timestamp_append_colon", None)

    out = fmt(0.23)
    rep = f"<{out}>" if wrapped else out
    rep = f"{rep}:" if appended else rep
    assert rep == "<0.2s>:"
