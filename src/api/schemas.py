from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator

class PredictRequest(BaseModel):
    features: List[float] = Field(
        ..., 
        description="List of exactly 48 numeric network flow features.",
        example=[0.1] * 48
    )

    @field_validator("features")
    @classmethod
    def validate_features_length(cls, v: List[float]) -> List[float]:
        if len(v) != 48:
            raise ValueError(f"Expected exactly 48 numeric features, got {len(v)}.")
        return v

class BatchPredictRequest(BaseModel):
    samples: List[PredictRequest] = Field(
        ...,
        description="List of sample objects, each containing exactly 48 numeric features."
    )

class PredictResponse(BaseModel):
    model_name: str = "DualScore-EdgeProto"
    model_version: str = "1.0.0"
    demo_mode: bool
    prediction: Union[int, str]
    nearest_known_class: Union[int, str]
    unknown_score: float
    zero_day: bool
    message: str

class BatchPredictResponse(BaseModel):
    model_name: str = "DualScore-EdgeProto"
    model_version: str = "1.0.0"
    demo_mode: bool
    predictions: List[PredictResponse]
    count: int

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "dualscore-edgeproto-api"
    demo_mode: bool
    version: str = "1.0.0"

class MetadataResponse(BaseModel):
    model_name: str = "DualScore-EdgeProto"
    model_version: str = "1.0.0"
    architecture: str = "Depthwise-Separable Conv1D Encoder + Metric Prototypes + 6-Layer Autoencoder"
    parameters: int = 15356
    expected_features: int = 48
    demo_mode: bool
    labels: List[str]
    threshold_zero_day: float = 0.0
