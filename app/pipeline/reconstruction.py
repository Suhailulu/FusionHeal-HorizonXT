from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelPrediction(BaseModel):
    available: bool = Field(
        False, description="True if a validated ML model inference was successfully executed"
    )
    candidate_value: Optional[float] = Field(
        None, description="Proposed point estimate for sensor reconstruction"
    )
    uncertainty_lower: Optional[float] = Field(
        None, description="Lower prediction interval bound (e.g. 5th percentile)"
    )
    uncertainty_upper: Optional[float] = Field(
        None, description="Upper prediction interval bound (e.g. 95th percentile)"
    )
    confidence_score: Optional[float] = Field(
        None, description="Calibrated confidence score between 0.0 and 1.0; null if unavailable"
    )
    model_name: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Semantic version of model weights / graph")
    reconstruction_method: Optional[str] = Field(None, description="Method used (e.g. saits_transformer)")
    notes: str = Field(..., description="Operational status and diagnostic context")


class ReconstructionModel(ABC):
    """Abstract Base Class for Spatial-Temporal ML Reconstruction Models.

    Teammates developing PyTorch / ONNX models (SAITS, ImTransformer, or Quantile BiLSTM)
    should implement this interface to plug into the FusionHeal pipeline.
    """

    @abstractmethod
    def predict(
        self,
        asset_id: str,
        sensor_id: str,
        sensor_window: List[Dict[str, Any]],
    ) -> ModelPrediction:
        """Accept a prepared time-series observation window.

        Args:
            asset_id: Asset identifier (e.g. boiler_01)
            sensor_id: Sensor channel identifier (e.g. temperature_01)
            sensor_window: Sequential list of historical observations including timestamps,
                           values, and missingness masks.

        Returns:
            ModelPrediction object with candidate value and uncertainty intervals.

        Note:
            The DecisionEngine must independently validate predictions against physical
            constraints before accepting any repairs.
        """
        raise NotImplementedError


class PlaceholderReconstructionAdapter(ReconstructionModel):
    """Explicit placeholder adapter for development and testing.

    CRITICAL ARCHITECTURE RULE:
    This placeholder adapter NEVER pretends to provide real ML predictions or meaningful
    confidence scores. When the Transformer model is not connected, inference is
    explicitly marked as unavailable.
    """

    def __init__(self, model_name: str = "PlaceholderAdapter", version: str = "0.0.0-unconnected"):
        self.model_name = model_name
        self.version = version

    def predict(
        self,
        asset_id: str,
        sensor_id: str,
        sensor_window: List[Dict[str, Any]],
    ) -> ModelPrediction:
        """Returns an explicit unavailable prediction state."""
        return ModelPrediction(
            available=False,
            candidate_value=None,
            uncertainty_lower=None,
            uncertainty_upper=None,
            confidence_score=None,
            model_name=self.model_name,
            model_version=self.version,
            reconstruction_method=None,
            notes=(
                "ML model is not connected in this phase. "
                "No synthetic candidate fabricated. Pipeline remains fully functional for "
                "ingestion, preservation, detection, and unresolved issue recording."
            ),
        )
