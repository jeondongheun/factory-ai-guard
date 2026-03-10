from fastapi import APIRouter, Query
from app.models.database import AsyncSessionLocal, DetectionResult, SensorReading, LLMDiagnosis
from sqlalchemy import select, desc, and_
from datetime import datetime
from typing import Optional

router = APIRouter()

@router.get("/")
async def get_history(
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    limit: int = 50,
    offset: int = 0
):
    """탐지 이력 조회 (필터링 지원)"""
    async with AsyncSessionLocal() as db:
        query = select(DetectionResult).order_by(desc(DetectionResult.timestamp))

        # 날짜 필터
        conditions = []
        if from_date:
            conditions.append(DetectionResult.timestamp >= datetime.fromisoformat(from_date))
        if to_date:
            conditions.append(DetectionResult.timestamp <= datetime.fromisoformat(to_date))
        if severity:
            conditions.append(DetectionResult.severity == severity)
        if conditions:
            query = query.where(and_(*conditions))

        result = await db.execute(query.limit(limit).offset(offset))
        detections = result.scalars().all()

        return {
            "items": [
                {
                    "id": d.id,
                    "timestamp": d.timestamp,
                    "anomaly_detected": d.anomaly_detected,
                    "probability": d.probability,
                    "severity": d.severity,
                }
                for d in detections
            ],
            "total": len(detections),
            "limit": limit,
            "offset": offset
        }
