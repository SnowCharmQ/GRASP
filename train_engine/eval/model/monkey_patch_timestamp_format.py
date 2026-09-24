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

# Qwen2.5-VL is optional; patch when available so callers can share the same API.
try:  # pragma: no cover - optional dependency
    from model.processing_qwen2_5_vl import (
        Qwen2_5_VLProcessor,
        Qwen2_5_VLProcessorKwargs,
    )
except Exception:  # pragma: no cover - qwen2.5 not installed
    Qwen2_5_VLProcessor = None
    Qwen2_5_VLProcessorKwargs = None


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


def _calculate_timestamps(indices: list[int], video_fps: float, merge_size: int = 2):
    """Reproduce the timestamp averaging logic used in the official processors.

    The raw frame indices are grouped by the temporal merge size and averaged to
    represent the merged frame's timestamp. This mirrors Qwen3VLProcessor._calculate_timestamps.
    """

    if not isinstance(indices, list):
        indices = list(indices)
    if len(indices) % merge_size != 0:
        indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
    timestamps = [idx / video_fps for idx in indices]
    return [
        (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
    ]


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
    """Monkey-patch Qwen3VLProcessor and (optionally) Qwen2_5_VLProcessor to support flexible timestamp text.

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

    def _update_class_attrs(processor_cls):
        processor_cls._timestamp_formatter = staticmethod(base_formatter)
        processor_cls._timestamp_wrap_brackets = base_wrap_brackets
        processor_cls._timestamp_append_colon = base_append_colon
        processor_cls._timestamp_use_timestamps = base_use_timestamps

    # ----------------
    # Patch Qwen3-VL
    # ----------------
    if getattr(Qwen3VLProcessor, "_timestamp_patch_installed", False):
        _update_class_attrs(Qwen3VLProcessor)
    else:
        Qwen3VLProcessor._timestamp_patch_installed = True
        _update_class_attrs(Qwen3VLProcessor)

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

    # ------------------
    # Patch Qwen2.5-VL
    # ------------------
    if Qwen2_5_VLProcessor is None:
        return

    if getattr(Qwen2_5_VLProcessor, "_timestamp_patch_installed", False):
        _update_class_attrs(Qwen2_5_VLProcessor)
        return

    Qwen2_5_VLProcessor._timestamp_patch_installed = True
    _update_class_attrs(Qwen2_5_VLProcessor)

    original_call_25 = Qwen2_5_VLProcessor.__call__

    def _patched_call_qwen25(self, images=None, text=None, videos=None, **kwargs):
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
        video_metadata = kwargs.pop("video_metadata", None)

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

        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        videos_inputs = {}
        video_grid_thw = None

        if videos is not None:
            try:
                videos_inputs = self.video_processor(
                    videos=videos,
                    video_metadata=video_metadata,
                    **output_kwargs["videos_kwargs"],
                )
            except TypeError:
                videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
                if video_metadata is not None and "video_metadata" not in videos_inputs:
                    videos_inputs["video_metadata"] = video_metadata

            video_grid_thw = videos_inputs["video_grid_thw"]
            if "video_metadata" in videos_inputs:
                video_metadata = videos_inputs.get("video_metadata")

        if not isinstance(text, list):
            text = [text]

        # Keep downstream parity with the original processor outputs
        second_per_grid_ts = []

        text = text.copy()
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
            merge_size = self.video_processor.merge_size
            temporal_patch = getattr(self.video_processor, "temporal_patch_size", 1)
            vision_start_token = getattr(self, "vision_start_token", "<|vision_start|>")
            vision_end_token = getattr(self, "vision_end_token", "<|vision_end|>")
            # Normalize metadata to list for easier indexing
            if video_metadata is not None and not isinstance(video_metadata, (list, tuple)):
                video_metadata = [video_metadata] * len(video_grid_thw)

            fps_arg = output_kwargs["videos_kwargs"].get("fps", 2.0)
            if isinstance(fps_arg, (int, float)):
                fps_per_video = [fps_arg] * len(video_grid_thw)
            elif hasattr(fps_arg, "__len__") and len(fps_arg) == len(video_grid_thw):
                fps_per_video = list(fps_arg)
            else:
                raise ValueError(
                    f"The length of fps ({len(fps_arg) if hasattr(fps_arg, '__len__') else fps_arg}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )

            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    metadata = video_metadata[index] if video_metadata is not None else None

                    per_video_fps = None
                    if metadata is not None and getattr(metadata, "fps", None):
                        per_video_fps = metadata.fps
                    else:
                        per_video_fps = fps_per_video[index]
                        if per_video_fps is None:
                            per_video_fps = 24.0

                    seconds_per_grid = temporal_patch / float(per_video_fps)
                    second_per_grid_ts.append(seconds_per_grid)

                    if metadata is not None and getattr(metadata, "frames_indices", None) is not None:
                        curr_timestamp = _calculate_timestamps(
                            metadata.frames_indices,
                            per_video_fps,
                            merge_size,
                        )
                    else:
                        synthetic_indices = list(range(int(video_grid_thw[index][0] * merge_size)))
                        curr_timestamp = _calculate_timestamps(
                            synthetic_indices,
                            per_video_fps,
                            merge_size,
                        )

                    # Qwen2.5's RoPE indexing expects exactly one <|vision_start|> block per video.
                    # So we DO NOT create multiple vision blocks (which would increase `video_nums` and
                    # desync with `video_grid_thw`). Instead we insert timestamp text between temporal
                    # groups of <|video_pad|> tokens while keeping a single vision wrapper.
                    t_grids = int(video_grid_thw[index][0])
                    frame_seqlen = int(video_grid_thw[index][1:].prod() // merge_length)

                    def _fmt_time(sec: float) -> str:
                        if not use_timestamps:
                            return ""
                        formatted = format_ts(sec)
                        if not formatted:
                            return ""
                        token = f"<{formatted}>" if effective_wrap else formatted
                        if effective_append:
                            token = f"{token}:"
                        return f"{token}"

                    # Timestamp for the first temporal slice must appear BEFORE <|vision_start|>, otherwise
                    # the token immediately after <|vision_start|> wouldn't be <|video_pad|> anymore.
                    t0 = curr_timestamp[0] if len(curr_timestamp) > 0 else 0.0
                    prefix_timestamp = _fmt_time(t0)

                    inner = ""
                    for frame_idx in range(t_grids):
                        # Token group for this temporal slice.
                        inner += "<|placeholder|>" * frame_seqlen

                        # Timestamp shown before the NEXT temporal slice.
                        if frame_idx + 1 < t_grids:
                            next_time = (
                                curr_timestamp[frame_idx + 1]
                                if (frame_idx + 1) < len(curr_timestamp)
                                else (frame_idx + 1) * seconds_per_grid
                            )
                            inner += _fmt_time(next_time)

                    replaced = False
                    wrapped_full = f"{vision_start_token}{self.video_token}{vision_end_token}"
                    if wrapped_full in text[i]:
                        replacement = f"{prefix_timestamp}{vision_start_token}{inner}{vision_end_token}"
                        text[i] = text[i].replace(wrapped_full, replacement, 1)
                        replaced = True
                    else:
                        wrapped_prefix = f"{vision_start_token}{self.video_token}"
                        if wrapped_prefix in text[i]:
                            replacement = f"{prefix_timestamp}{vision_start_token}{inner}"
                            text[i] = text[i].replace(wrapped_prefix, replacement, 1)
                            replaced = True

                    if not replaced:
                        # Fallback when the chat template doesn't include vision wrappers.
                        # Keep the original single-video semantics by NOT inserting any extra vision tokens.
                        text[i] = text[i].replace(self.video_token, f"{prefix_timestamp}{inner}", 1)
                    index += 1

                text[i] = text[i].replace("<|placeholder|>", self.video_token)

            videos_inputs["second_per_grid_ts"] = second_per_grid_ts

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

    Qwen2_5_VLProcessor.__call__ = _patched_call_qwen25
    Qwen2_5_VLProcessor._timestamp_original_call = original_call_25


__all__ = ["apply_timestamp_format_patch", "build_timestamp_formatter", "parse_ts_prompt"]