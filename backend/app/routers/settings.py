from fastapi import APIRouter, HTTPException
from app.models.database import AsyncSessionLocal, ThresholdSetting
from app.schemas.schemas import ThresholdUpdate
from sqlalchemy import select
from datetime import datetime

router = APIRouter()

@router.get("/thresholds")
async def get_thresholds():
    """임계값 설정 전체 조회"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(ThresholdSetting))
        settings = result.scalars().all()

        return {
            "thresholds": [
                {
                    "id": s.id,
                    "sensor_name": s.sensor_name,
                    "warning_value": s.warning_value,
                    "critical_value": s.critical_value,
                    "updated_at": s.updated_at
                }
                for s in settings
            ]
        }

@router.put("/thresholds")
async def update_threshold(body: ThresholdUpdate):
    """임계값 수정"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ThresholdSetting)
            .where(ThresholdSetting.sensor_name == body.sensor_name)
        )
        setting = result.scalar_one_or_none()

        if not setting:
            raise HTTPException(status_code=404, detail="센서를 찾을 수 없습니다")

        setting.warning_value = body.warning_value
        setting.critical_value = body.critical_value
        setting.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(setting)

        return {
            "sensor_name": setting.sensor_name,
            "warning_value": setting.warning_value,
            "critical_value": setting.critical_value,
            "updated_at": setting.updated_at
        }
