from fastapi import APIRouter, Request, HTTPException
from app.schemas.schemas import DetectionRequest, DetectionResponse
from app.models.database import AsyncSessionLocal, DetectionResult, SensorReading
from sqlalchemy import select, desc
from datetime import datetime

router = APIRouter()

@router.post("/analyze", response_model=DetectionResponse)
async def analyze(request: Request, body: DetectionRequest):
    """센서 윈도우 데이터 이상 탐지"""
    ml_service = request.app.state.ml_service
    result = ml_service.predict(body.sensor_data)

    # DB 저장
    async with AsyncSessionLocal() as db:
        # 최신 센서값 저장
        latest = body.sensor_data[-1]
        sensor = SensorReading(
            accelerometer1=latest[0],
            accelerometer2=latest[1],
            current=latest[2],
            pressure=latest[3],
            temperature=latest[4],
            thermocouple=latest[5],
            voltage=latest[6],
            flow_rate=latest[7],
        )
        db.add(sensor)
        await db.flush()

        # 탐지 결과 저장
        detection = DetectionResult(
            anomaly_detected=result["anomaly_detected"],
            probability=result["probability"],
            severity=result["severity"],
            sensor_reading_id=sensor.id
        )
        db.add(detection)
        await db.commit()
        await db.refresh(detection)

        return DetectionResponse(
            anomaly_detected=result["anomaly_detected"],
            probability=result["probability"],
            severity=result["severity"],
            sensor_reading_id=sensor.id,
            # RAAD-LLM 확장 필드
            mode=result.get("mode"),
            z_score=result.get("z_score"),
            primary_sensor=result.get("primary_sensor"),
            fault_type=result.get("fault_type"),
        )

@router.get("/history")
async def get_detection_history(limit: int = 50, offset: int = 0):
    """탐지 이력 조회"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(DetectionResult)
            .order_by(desc(DetectionResult.timestamp))
            .limit(limit)
            .offset(offset)
        )
        detections = result.scalars().all()

        return {
            "items": [
                {
                    "id": d.id,
                    "timestamp": d.timestamp,
                    "anomaly_detected": d.anomaly_detected,
                    "probability": d.probability,
                    "severity": d.severity,
                    "sensor_reading_id": d.sensor_reading_id
                }
                for d in detections
            ],
            "total": len(detections),
            "limit": limit,
            "offset": offset
        }

@router.get("/{detection_id}")
async def get_detection(detection_id: int):
    """특정 탐지 결과 조회"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(DetectionResult)
            .where(DetectionResult.id == detection_id)
        )
        detection = result.scalar_one_or_none()

        if not detection:
            raise HTTPException(status_code=404, detail="탐지 결과를 찾을 수 없습니다")

        return {
            "id": detection.id,
            "timestamp": detection.timestamp,
            "anomaly_detected": detection.anomaly_detected,
            "probability": detection.probability,
            "severity": detection.severity,
            "sensor_reading_id": detection.sensor_reading_id
        }
