from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from app.models.database import AsyncSessionLocal, DetectionResult, SensorReading
import pandas as pd
import numpy as np
import io
import uuid

router = APIRouter()

# 분석 작업 임시 저장 (실제 서비스라면 Redis 사용)
job_results = {}

@router.post("/csv")
async def upload_csv(request: Request, file: UploadFile = File(...)):
    """CSV 파일 배치 분석"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="CSV 파일만 업로드 가능합니다")

    ml_service = request.app.state.ml_service

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')), sep=None, engine='python')

    feature_cols = [
        'Accelerometer1RMS', 'Accelerometer2RMS',
        'Current', 'Pressure', 'Temperature',
        'Thermocouple', 'Voltage', 'Volume Flow RateRMS'
    ]

    # 컬럼 매핑
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < 4:
        raise HTTPException(status_code=400, detail="센서 컬럼이 부족합니다")

    features = df[available].fillna(0).values
    window_size = 60
    results = []

    for i in range(0, len(features) - window_size, 10):
        window = features[i:i + window_size].tolist()
        result = ml_service.predict(window)
        results.append({
            "index": i,
            "anomaly_detected": result["anomaly_detected"],
            "probability": result["probability"],
            "severity": result["severity"]
        })

    # 요약 통계
    total = len(results)
    anomalies = sum(1 for r in results if r["anomaly_detected"])

    job_id = str(uuid.uuid4())[:8]
    job_results[job_id] = {
        "filename": file.filename,
        "total_windows": total,
        "anomaly_count": anomalies,
        "anomaly_rate": round(anomalies / total, 4) if total > 0 else 0,
        "results": results
    }

    return {"job_id": job_id, "total_windows": total, "anomaly_count": anomalies}

@router.get("/result/{job_id}")
async def get_result(job_id: str):
    """배치 분석 결과 조회"""
    if job_id not in job_results:
        raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다")
    return job_results[job_id]
