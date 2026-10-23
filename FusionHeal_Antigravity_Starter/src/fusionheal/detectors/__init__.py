"""FusionHeal Fault Detection Module.

Contains modular, evidence-guided detectors for industrial time-series sensor anomalies.
"""

from .base import BaseDetector, format_issue
from .missing import MissingValueDetector
from .range_check import RangeViolationDetector
from .outlier import RobustOutlierDetector
from .spike import SpikeDetector
from .stuck import StuckSensorDetector
from .drift import DriftDetector
from .timestamp import TimestampIntegrityDetector
from .cross_sensor import CrossSensorConsistencyDetector

__all__ = [
    "BaseDetector",
    "format_issue",
    "MissingValueDetector",
    "RangeViolationDetector",
    "RobustOutlierDetector",
    "SpikeDetector",
    "StuckSensorDetector",
    "DriftDetector",
    "TimestampIntegrityDetector",
    "CrossSensorConsistencyDetector",
]
