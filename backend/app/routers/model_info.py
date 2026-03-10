from fastapi import APIRouter, Request
from app.models.database import AsyncSessionLocal, ModelMetric
from sqlalchemy import select, desc

router = APIRouter()

@router.get("/metrics")
async def get_metrics(request: Request):
    """모델 성능 지표 반환"""
    ml_service = request.app.state.ml_service
    metrics = ml_service.get_metrics()

    return {
        "accuracy":  round(metrics.get("accuracy", 0), 4),
        "precision": round(metrics.get("precision", 0), 4),
        "recall":    round(metrics.get("recall", 0), 4),
        "f1":        round(metrics.get("f1", 0), 4),
        "model_loaded": ml_service.is_loaded
    }

@router.get("/info")
async def get_model_info(request: Request):
    """모델 구조 정보"""
    ml_service = request.app.state.ml_service

    return {
        "model_name": "LSTM TimeSeriesEncoder",
        "input_dim": ml_service.input_dim,
        "hidden_dim": 128,
        "num_layers": 2,
        "window_size": ml_service.window_size,
        "device": ml_service.device,
        "is_loaded": ml_service.is_loaded,
        "model_path": str(ml_service.model_path)
    }
