from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch

from app.routers import sensors, detection, diagnosis, history, stats, model_info, upload, settings
from app.services.ml_service import MLService
from app.services.sensor_simulator import SensorSimulator
from app.models.database import engine, Base

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행
    print("🚀 FactoryGuard-AI 서버 시작")

    # DB 테이블 생성
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # ML 모델 로드
    app.state.ml_service = MLService()
    app.state.ml_service.load_model()

    # 센서 시뮬레이터 시작
    app.state.simulator = SensorSimulator()

    print("✅ 초기화 완료")
    yield

    # 종료 시 실행
    print("👋 서버 종료")

app = FastAPI(
    title="FactoryGuard-AI",
    description="Real-time smart factory anomaly detection system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(sensors.router, prefix="/api/sensors", tags=["sensors"])
app.include_router(detection.router, prefix="/api/detection", tags=["detection"])
app.include_router(diagnosis.router, prefix="/api/diagnosis", tags=["diagnosis"])
app.include_router(history.router, prefix="/api/history", tags=["history"])
app.include_router(stats.router, prefix="/api/stats", tags=["stats"])
app.include_router(model_info.router, prefix="/api/model", tags=["model"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])

@app.get("/")
async def root():
    return {"message": "FactoryGuard-AI API", "docs": "/docs"}
