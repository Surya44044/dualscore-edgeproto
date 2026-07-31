import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import (
    PredictRequest, PredictResponse, 
    BatchPredictRequest, BatchPredictResponse,
    HealthResponse, MetadataResponse
)
from src.api.preprocess import preprocess_features, preprocess_batch_features
from src.api.model_loader import model_manager

app = FastAPI(
    title="DualScore-EdgeProto Inference API",
    description="Lightweight Open-Set Zero-Day IoT Intrusion Detection Inference Microservice",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "dualscore-edgeproto-api",
        "message": "DualScore-EdgeProto API is running. Use /health, /metadata, /docs, /predict, or /predict_batch.",
        "docs": "/docs",
        "health": "/health",
        "metadata": "/metadata",
        "demo_mode": model_manager.demo_mode
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        service="dualscore-edgeproto-api",
        demo_mode=model_manager.demo_mode,
        version="1.0.0"
    )

@app.get("/metadata", response_model=MetadataResponse)
def get_metadata():
    return MetadataResponse(
        model_name="DualScore-EdgeProto",
        model_version="1.0.0",
        architecture="Depthwise-Separable Conv1D Encoder + Metric Prototypes + 6-Layer Autoencoder",
        parameters=15356,
        expected_features=48,
        demo_mode=model_manager.demo_mode,
        labels=model_manager.label_map,
        threshold_zero_day=model_manager.zero_day_threshold
    )

@app.post("/predict", response_model=PredictResponse)
def predict_single(request: PredictRequest):
    try:
        x_tensor = preprocess_features(request.features)
        res = model_manager.predict(x_tensor)
        
        if isinstance(res, dict):
            return PredictResponse(
                model_name="DualScore-EdgeProto",
                model_version="1.0.0",
                demo_mode=res["demo_mode"],
                prediction=res["prediction"],
                nearest_known_class=res["nearest_known_class"],
                unknown_score=res["unknown_score"],
                zero_day=res["zero_day"],
                message=res["message"]
            )
        else:
            raise HTTPException(status_code=500, detail="Unexpected prediction result structure.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    try:
        raw_samples = [s.features for s in request.samples]
        x_tensor = preprocess_batch_features(raw_samples)
        res_list = model_manager.predict(x_tensor)
        
        if not isinstance(res_list, list):
            res_list = [res_list]
            
        responses = []
        for r in res_list:
            responses.append(PredictResponse(
                model_name="DualScore-EdgeProto",
                model_version="1.0.0",
                demo_mode=r["demo_mode"],
                prediction=r["prediction"],
                nearest_known_class=r["nearest_known_class"],
                unknown_score=r["unknown_score"],
                zero_day=r["zero_day"],
                message=r["message"]
            ))
            
        return BatchPredictResponse(
            model_name="DualScore-EdgeProto",
            model_version="1.0.0",
            demo_mode=model_manager.demo_mode,
            predictions=responses,
            count=len(responses)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")
