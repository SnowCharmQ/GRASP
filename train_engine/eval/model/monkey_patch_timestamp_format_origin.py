"""Monkey patch to customize timestamp formatting in Qwen3VLProcessor prompts.

Usage
-----
from model.monkey_patch_timestamp_format import apply_timestamp_format_patch

# Apply once at startup
apply_timestamp_format_patch(ts_prompt="seconds;unit=1;brackets=0;append_colon=1")

# Optionally override per call via `timestamp_prompt` kwargs when invoking the processor.
"""

from __future__ import annotations

from typing import Callable, Optional

import re

import numpy as np

from model.processing_qwen3_vl import (
    BatchFeature,
    Qwen3VLProcessor,
    Qwen3VLProcessorKwargs,
    logger,
)


def _format_seconds(seconds: float, include_unit: bool, decimal_places: int) -> str:
    rounded = round(seconds, decimal_places)
    suffix = "s" if include_unit else ""
    return f"{rounded}{suffix}"


def _format_integer_seconds(seconds: float, include_unit: bool) -> str:
    secs = int(round(seconds))
    suffix = "s" if include_unit else ""
    return f"{secs}{suffix}"


def _format_seconds_word(seconds: float, include_unit: bool, decimal_places: int) -> str:
    rounded = round(seconds, decimal_places)
    suffix = " seconds" if include_unit else ""
    return f"{rounded}{suffix}"


def _format_hms(seconds: float, trailing_colon: bool) -> str:
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}{':' if trailing_colon else ''}"


def _format_min_sec(seconds: float) -> str:
    total = int(round(seconds))
    m = total // 60
    s = total % 60
    return f"{m}min{s:02d}s"


def _noop_timestamp(_: float) -> str:
    return ""


def build_timestamp_formatter(
    style: str = "seconds",
    *,
    include_unit: bool = True,
    decimal_places: int = 1,
    trailing_colon: bool = False,
) -> Callable[[float], str]:
    style = (style or "seconds").lower()

    if style in {"none", "off", "disable", "disabled", "no_ts", "no-timestamp", "notimestamps"}:
        return _noop_timestamp

    if style in {"seconds", "s_float"}:
        return lambda x: _format_seconds(x, include_unit=include_unit, decimal_places=decimal_places)
    if style in {"seconds_word", "seconds-word", "seconds_words"}:
        return lambda x: _format_seconds_word(x, include_unit=include_unit, decimal_places=decimal_places)
    if style in {"s", "ss", "seconds_int"}:
        return lambda x: _format_integer_seconds(x, include_unit=include_unit)
    if style in {"hms", "hh:mm:ss"}:
        return lambda x: _format_hms(x, trailing_colon=trailing_colon)
    if style in {"hms_colon"}:
        return lambda x: _format_hms(x, trailing_colon=True)
    if style in {"minsec", "mmss", "minutes_seconds"}:
        return _format_min_sec

    # Fallback to default seconds with unit
    return lambda x: _format_seconds(x, include_unit=True, decimal_places=decimal_places)


def _parse_bool(val: str, default: bool) -> bool:
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "y"}


def parse_ts_prompt(prompt: Optional[str]) -> dict:
    """Parse a compact timestamp prompt string into formatting options.

    Syntax example: "seconds;unit=1;brackets=0;append_colon=1;decimals=1;trailing_colon=0"
    Rules:
      - Items are separated by ';' or ','.
      - A lone token without '=' is treated as style (e.g., "seconds" or "hms").
      - Boolean-like values accept 1/0/true/false/yes/no/y/n (case-insensitive).
            - Recognized keys: style, unit/include_unit, decimals, trailing_colon,
                brackets/wrap_with_brackets, append_colon, use_timestamps/disable_timestamps.
    """

    if not prompt:
        return {}

    parts = [p.strip() for p in re.split(r"[;,]", prompt) if p.strip()]
    opts: dict = {}
    for part in parts:
        if "=" not in part:
            opts["style"] = part.lower()
            continue
        key, val = part.split("=", 1)
        key = key.strip().lower()
        val = val.strip()
        if key in {"style"}:
            opts["style"] = val.lower()
        elif key in {"unit", "include_unit"}:
            opts["include_unit"] = _parse_bool(val, True)
        elif key in {"decimals", "decimal", "precision"}:
            try:
                opts["decimal_places"] = int(val)
            except ValueError:
                pass
        elif key in {"trailing_colon"}:
            opts["trailing_colon"] = _parse_bool(val, False)
        elif key in {"brackets", "wrap", "wrap_with_brackets"}:
            opts["wrap_with_brackets"] = _parse_bool(val, True)
        elif key in {"append_colon", "colon"}:
            opts["append_colon"] = _parse_bool(val, False)
        elif key in {"use_timestamps", "timestamps", "enable_timestamps"}:
            opts["use_timestamps"] = _parse_bool(val, True)
        elif key in {"disable_timestamps", "no_timestamps", "no_ts", "disable_ts"}:
            opts["use_timestamps"] = not _parse_bool(val, True)
    return opts


def apply_timestamp_format_patch(
    *,
    ts_prompt: Optional[str] = None,
    format_style: Optional[str] = None,
    include_unit: Optional[bool] = None,
    decimal_places: Optional[int] = None,
    trailing_colon: Optional[bool] = None,
    wrap_with_brackets: Optional[bool] = None,
    append_colon: Optional[bool] = None,
    formatter: Optional[Callable[[float], str]] = None,
):
    """Monkey-patch Qwen3VLProcessor.__call__ to support multiple timestamp formats.

    Priority of formatter selection:
    1) Explicit `formatter` callable passed here.
    2) `ts_prompt` string (compact rule expression) passed here.
    3) `format_style` and related kwargs passed here.
    4) Defaults (seconds with unit, 1 decimal, bracketed, no trailing colon/append colon).
    """
    prompt_opts = parse_ts_prompt(ts_prompt)
    chosen_style = prompt_opts.get("style", format_style or "seconds")
    chosen_include_unit = prompt_opts.get("include_unit", True if include_unit is None else include_unit)
    chosen_decimals = prompt_opts.get("decimal_places", 1 if decimal_places is None else decimal_places)
    chosen_trailing = prompt_opts.get("trailing_colon", False if trailing_colon is None else trailing_colon)
    base_wrap_brackets = prompt_opts.get("wrap_with_brackets", True if wrap_with_brackets is None else wrap_with_brackets)
    base_append_colon = prompt_opts.get("append_colon", False if append_colon is None else append_colon)

    no_ts_styles = {"none", "off", "disable", "disabled", "no_ts", "no-timestamp", "notimestamps"}
    prompt_use_ts = prompt_opts.get("use_timestamps")
    base_use_timestamps = (prompt_use_ts if prompt_use_ts is not None else chosen_style not in no_ts_styles)
    if not base_use_timestamps:
        base_wrap_brackets = False
        base_append_colon = False

    base_formatter = formatter or build_timestamp_formatter(
        style=chosen_style,
        include_unit=chosen_include_unit,
        decimal_places=chosen_decimals,
        trailing_colon=chosen_trailing,
    )

    if getattr(Qwen3VLProcessor, "_timestamp_patch_installed", False):
        # staticmethod avoids descriptor binding that would inject `self`
        Qwen3VLProcessor._timestamp_formatter = staticmethod(base_formatter)
        Qwen3VLProcessor._timestamp_wrap_brackets = base_wrap_brackets
        Qwen3VLProcessor._timestamp_append_colon = base_append_colon
        Qwen3VLProcessor._timestamp_use_timestamps = base_use_timestamps
        return

    Qwen3VLProcessor._timestamp_patch_installed = True
    Qwen3VLProcessor._timestamp_formatter = staticmethod(base_formatter)
    Qwen3VLProcessor._timestamp_wrap_brackets = base_wrap_brackets
    Qwen3VLProcessor._timestamp_append_colon = base_append_colon
    Qwen3VLProcessor._timestamp_use_timestamps = base_use_timestamps

    original_call = Qwen3VLProcessor.__call__

    def _patched_call(self, images=None, text=None, videos=None, **kwargs):
        # Per-call overrides (popped to avoid leaking into _merge_kwargs).
        per_call_prompt = kwargs.pop("timestamp_prompt", None)
        per_call_style = kwargs.pop("timestamp_format", None)
        per_call_include_unit = kwargs.pop("timestamp_include_unit", None)
        per_call_decimals = kwargs.pop("timestamp_decimals", None)
        per_call_trailing_colon = kwargs.pop("timestamp_trailing_colon", None)
        per_call_wrap_brackets = kwargs.pop("timestamp_wrap_brackets", None)
        per_call_append_colon = kwargs.pop("timestamp_append_colon", None)
        per_call_formatter = kwargs.pop("timestamp_formatter", None)
        per_call_use_ts = kwargs.pop("timestamp_enable", None)
        per_call_disable_ts = kwargs.pop("timestamp_disable", None)

        format_ts = per_call_formatter or getattr(self.__class__, "_timestamp_formatter", base_formatter)
        prompt_overrides = parse_ts_prompt(per_call_prompt)

        if prompt_overrides or per_call_style or per_call_include_unit is not None or per_call_decimals is not None or per_call_trailing_colon is not None:
            format_ts = build_timestamp_formatter(
                style=prompt_overrides.get("style", per_call_style or chosen_style),
                include_unit=prompt_overrides.get(
                    "include_unit",
                    per_call_include_unit if per_call_include_unit is not None else chosen_include_unit,
                ),
                decimal_places=prompt_overrides.get(
                    "decimal_places",
                    per_call_decimals if per_call_decimals is not None else chosen_decimals,
                ),
                trailing_colon=prompt_overrides.get(
                    "trailing_colon",
                    per_call_trailing_colon if per_call_trailing_colon is not None else chosen_trailing,
                ),
            )

        no_ts_style = prompt_overrides.get("style", chosen_style) in no_ts_styles
        use_timestamps = getattr(self.__class__, "_timestamp_use_timestamps", base_use_timestamps)
        if per_call_use_ts is not None:
            use_timestamps = bool(per_call_use_ts)
        if per_call_disable_ts is not None:
            use_timestamps = not bool(per_call_disable_ts)
        if "use_timestamps" in prompt_overrides:
            use_timestamps = bool(prompt_overrides["use_timestamps"])
        if no_ts_style:
            use_timestamps = False

        use_brackets = (
            prompt_overrides.get("wrap_with_brackets")
            if "wrap_with_brackets" in prompt_overrides
            else (
                per_call_wrap_brackets
                if per_call_wrap_brackets is not None
                else getattr(self.__class__, "_timestamp_wrap_brackets", base_wrap_brackets)
            )
        )
        use_append_colon = (
            prompt_overrides.get("append_colon")
            if "append_colon" in prompt_overrides
            else (
                per_call_append_colon
                if per_call_append_colon is not None
                else getattr(self.__class__, "_timestamp_append_colon", base_append_colon)
            )
        )

        effective_wrap = use_brackets and use_timestamps
        effective_append = use_append_colon and use_timestamps

        # The body mirrors the original __call__, except timestamp formatting is delegated to `format_ts`.
        output_kwargs = self._merge_kwargs(
            Qwen3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
            # If user has not requested video metadata, pop it
            if "return_metadata" not in kwargs:
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        logger.warning_once(
                            "Qwen3VL requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                            "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                            "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                        )
                        metadata.fps = 24 if metadata.fps is None else metadata.fps

                    curr_timestamp = self._calculate_timestamps(
                        metadata.frames_indices,
                        metadata.fps,
                        self.video_processor.merge_size,
                    )

                    video_placeholder = ""
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
                    for frame_idx in range(video_grid_thw[index][0]):
                        curr_time = curr_timestamp[frame_idx]
                        if use_timestamps:
                            formatted_time = format_ts(curr_time)
                            if formatted_time:
                                timestamp_repr = f"<{formatted_time}>" if effective_wrap else formatted_time
                            else:
                                timestamp_repr = ""
                            if effective_append and timestamp_repr:
                                timestamp_repr = f"{timestamp_repr}:"
                            video_placeholder += timestamp_repr
                        video_placeholder += (
                            self.vision_start_token + "<|placeholder|>" * frame_seqlen + self.vision_end_token
                        )
                    if f"{self.vision_start_token}{self.video_token}{self.vision_end_token}" in text[i]:
                        text[i] = text[i].replace(
                            f"{self.vision_start_token}{self.video_token}{self.vision_end_token}", video_placeholder, 1
                        )
                    else:
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    index += 1

                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

    Qwen3VLProcessor.__call__ = _patched_call
    Qwen3VLProcessor._timestamp_original_call = original_call


__all__ = ["apply_timestamp_format_patch", "build_timestamp_formatter", "parse_ts_prompt"]