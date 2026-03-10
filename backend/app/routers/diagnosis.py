from fastapi import APIRouter, Request, HTTPException
from app.models.database import AsyncSessionLocal, LLMDiagnosis, DetectionResult
from app.services.llm_service import LLMService
from sqlalchemy import select

router = APIRouter()
llm_service = LLMService()

@router.post("/llm")
async def llm_diagnose(request: Request, body: dict):
    """LLM 기반 상세 진단"""
    detection_id = body.get("detection_id")
    sensor_stats = body.get("sensor_stats", {})

    async with AsyncSessionLocal() as db:
        # 탐지 결과 조회
        result = await db.execute(
            select(DetectionResult)
            .where(DetectionResult.id == detection_id)
        )
        detection = result.scalar_one_or_none()

        if not detection:
            raise HTTPException(status_code=404, detail="탐지 결과 없음")

        # LLM 진단 요청
        diagnosis = await llm_service.diagnose(
            probability=detection.probability,
            severity=detection.severity,
            sensor_stats=sensor_stats
        )

        # DB 저장
        llm_record = LLMDiagnosis(
            detection_id=detection_id,
            probable_cause=diagnosis["probable_cause"],
            recommendation=diagnosis["recommendation"]
        )
        db.add(llm_record)
        await db.commit()
        await db.refresh(llm_record)

        return {
            "id": llm_record.id,
            "detection_id": detection_id,
            "probable_cause": diagnosis["probable_cause"],
            "recommendation": diagnosis["recommendation"],
            "created_at": llm_record.created_at
        }

@router.get("/{detection_id}")
async def get_diagnosis(detection_id: int):
    """저장된 진단 결과 조회"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(LLMDiagnosis)
            .where(LLMDiagnosis.detection_id == detection_id)
            .order_by(LLMDiagnosis.created_at.desc())
        )
        diagnosis = result.scalar_one_or_none()

        if not diagnosis:
            raise HTTPException(status_code=404, detail="진단 결과 없음")

        return {
            "id": diagnosis.id,
            "detection_id": diagnosis.detection_id,
            "probable_cause": diagnosis.probable_cause,
            "recommendation": diagnosis.recommendation,
            "created_at": diagnosis.created_at
        }
