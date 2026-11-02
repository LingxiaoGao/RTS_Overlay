"""Apply confirmed HUD time anchors to the existing build-order timer."""

from __future__ import annotations


def anchor_build_order_timer(timer: dict, seconds: int, now: float) -> None:
    timer['use_timer'] = True
    timer['run_timer'] = True
    timer['time_sec'] = float(seconds)
    timer['time_int'] = seconds
    timer['time_sec_init'] = float(seconds)
    timer['absolute_time_init'] = float(now)
    timer['last_time_int'] = seconds
