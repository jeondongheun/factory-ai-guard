from fastapi import APIRouter
from app.models.database import AsyncSessionLocal, DetectionResult, SensorReading
from sqlalchemy import select, func, desc
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/summary")
async def get_summary():
    """전체 통계 요약"""
    async with AsyncSessionLocal() as db:
        # 전체 탐지 수
        total = await db.execute(select(func.count(DetectionResult.id)))
        total_count = total.scalar()

        # 이상 탐지 수
        anomaly = await db.execute(
            select(func.count(DetectionResult.id))
            .where(DetectionResult.anomaly_detected == True)
        )
        anomaly_count = anomaly.scalar()

        # 평균 이상 확률
        avg_prob = await db.execute(select(func.avg(DetectionResult.probability)))
        avg_probability = avg_prob.scalar() or 0.0

        return {
            "total_readings": total_count,
            "total_anomalies": anomaly_count,
            "anomaly_rate": round(anomaly_count / total_count, 4) if total_count > 0 else 0,
            "avg_probability": round(float(avg_probability), 4)
        }

@router.get("/trend")
async def get_trend(days: int = 7):
    """기간별 이상 발생 트렌드"""
    async with AsyncSessionLocal() as db:
        since = datetime.utcnow() - timedelta(days=days)
        result = await db.execute(
            select(DetectionResult)
            .where(DetectionResult.timestamp >= since)
            .order_by(DetectionResult.timestamp)
        )
        detections = result.scalars().all()

        # 일별 집계
        trend = {}
        for d in detections:
            day = d.timestamp.strftime("%Y-%m-%d")
            if day not in trend:
                trend[day] = {"total": 0, "anomalies": 0}
            trend[day]["total"] += 1
            if d.anomaly_detected:
                trend[day]["anomalies"] += 1

        return {
            "trend": [
                {"date": k, **v}
                for k, v in sorted(trend.items())
            ]
        }

@router.get("/sensor-avg")
async def get_sensor_avg():
    """센서별 평균값"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(
                func.avg(SensorReading.temperature).label("temperature"),
                func.avg(SensorReading.accelerometer1).label("accelerometer1"),
                func.avg(SensorReading.accelerometer2).label("accelerometer2"),
                func.avg(SensorReading.pressure).label("pressure"),
                func.avg(SensorReading.flow_rate).label("flow_rate"),
                func.avg(SensorReading.current).label("current"),
                func.avg(SensorReading.voltage).label("voltage"),
            )
        )
        row = result.one()

        return {
            "temperature":    round(float(row.temperature or 0), 2),
            "accelerometer1": round(float(row.accelerometer1 or 0), 4),
            "accelerometer2": round(float(row.accelerometer2 or 0), 4),
            "pressure":       round(float(row.pressure or 0), 3),
            "flow_rate":      round(float(row.flow_rate or 0), 3),
            "current":        round(float(row.current or 0), 3),
            "voltage":        round(float(row.voltage or 0), 2),
        }
