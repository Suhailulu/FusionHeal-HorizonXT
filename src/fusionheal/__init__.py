"""FusionHeal: Evidence-guided industrial data-quality and self-healing pipeline."""

__version__ = "0.2.0"

from .evidence import EvidenceEngine
from .reconstruction import ReconstructionEngine, generate_candidate
from .constraints import ProcessConstraintEngine
from .confidence import ConfidenceEngine
from .decisions import DecisionEngine, decide
from .detection import FaultDetectionEngine, detect_issues
from .pipeline import run_pipeline

__all__ = [
    "EvidenceEngine",
    "ReconstructionEngine",
    "ProcessConstraintEngine",
    "ConfidenceEngine",
    "DecisionEngine",
    "FaultDetectionEngine",
    "generate_candidate",
    "decide",
    "detect_issues",
    "run_pipeline",
]
