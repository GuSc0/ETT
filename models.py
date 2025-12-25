"""
Data models and constants for the Eye Tracking Tool.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List


EXPECTED_COLUMNS: List[str] = [
    "Recording", "Participant", "TOI", "Interval", "Bin_duration", "Media", "Event_type",
    "Validity", "EventIndex", "Start", "Stop", "Start_bin", "Stop_bin", "Duration",
    "AOI", "Hit_proportion", "FixationPointX", "FixationPointY", "Average_pupil_diameter",
    "Saccade_direction", "Average_velocity", "Peak_velocity", "Saccade_amplitude",
    "Start_AOI", "Landing_AOI", "Start_position_X", "Start_position_Y",
    "Landing_position_X", "Landing_position_Y", "Glance_AOI", "Glance_previous_AOI",
    "Glance_next_AOI",
]


PARAMETER_OPTIONS: List[str] = [
    "Task Completion Time (TCT)",
    "Standard Deviation of TCT",
    "Pupil Diameter",
    "Saccade Velocity",
    "Peak Saccade Velocity",
    "Saccade Amplitude",
]


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    message: Optional[str] = None
    found_columns: Optional[List[str]] = None
