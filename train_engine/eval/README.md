Here’s how each option affects the rendered timestamp (assume the raw time is 0.23 seconds unless noted):

style

seconds / s_float: uses decimals → with defaults 0.2s
s / ss / seconds_int: integer seconds → 0s
hms / hh:mm:ss: 00:00:00
hms_colon: 00:00:00:
minsec / mmss / minutes_seconds: 0min00s
include_unit (unit, include_unit)

include_unit=1 with seconds: 0.2s
include_unit=0 with seconds: 0.2
decimal_places (decimals, decimal, precision) — only for float seconds

decimals=1: 0.2s
decimals=3: 0.230s
trailing_colon

With hms and trailing_colon=1: 00:00:00:
With seconds and trailing_colon=1: 0.2s: (if no append_colon override)
wrap_with_brackets (brackets, wrap)

wrap_with_brackets=1: <0.2s>
wrap_with_brackets=0: 0.2s
append_colon

append_colon=1 + brackets on: <0.2s>:
append_colon=1 + brackets off: 0.2s:
append_colon=0: leaves as-is
Putting it together (with seconds style):

seconds;unit=1;brackets=1;append_colon=0;decimals=1 → <0.2s>
seconds;unit=1;brackets=0;append_colon=1;decimals=1 → 0.2s:
seconds;unit=0;brackets=0;append_colon=0;decimals=3 → 0.230
seconds;unit=1;brackets=1;append_colon=1 → <0.2s>:
With hms style (raw time 3661.2s):

hms;brackets=1;append_colon=0;trailing_colon=0 → <01:01:01>
hms;brackets=0;append_colon=1;trailing_colon=0 → 01:01:01:
hms;brackets=0;append_colon=0;trailing_colon=1 → 01:01:01:
How to pass:

Global: apply_timestamp_format_patch(ts_prompt="seconds;unit=1;brackets=0;append_colon=1;decimals=1")
Per-call: processor(..., timestamp_prompt="hms;brackets=1;append_colon=1")