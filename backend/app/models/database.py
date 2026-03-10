from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, Float, Boolean, String, DateTime, Text, ForeignKey
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = (
    f"postgresql+asyncpg://"
    f"{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}"
    f"/{os.getenv('POSTGRES_DB')}"
)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# DB 세션 의존성
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# ORM 모델
class SensorReading(Base):
    __tablename__ = "sensor_readings"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    accelerometer1 = Column(Float)
    accelerometer2 = Column(Float)
    current = Column(Float)
    pressure = Column(Float)
    temperature = Column(Float)
    thermocouple = Column(Float)
    voltage = Column(Float)
    flow_rate = Column(Float)

class DetectionResult(Base):
    __tablename__ = "detection_results"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    anomaly_detected = Column(Boolean)
    probability = Column(Float)
    severity = Column(String(10))
    sensor_reading_id = Column(Integer, ForeignKey("sensor_readings.id"))

class LLMDiagnosis(Base):
    __tablename__ = "llm_diagnoses"
    id = Column(Integer, primary_key=True)
    detection_id = Column(Integer, ForeignKey("detection_results.id"))
    probable_cause = Column(Text)
    recommendation = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class ThresholdSetting(Base):
    __tablename__ = "threshold_settings"
    id = Column(Integer, primary_key=True)
    sensor_name = Column(String(50), unique=True)
    warning_value = Column(Float)
    critical_value = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow)

class ModelMetric(Base):
    __tablename__ = "model_metrics"
    id = Column(Integer, primary_key=True)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1 = Column(Float)
    recorded_at = Column(DateTime, default=datetime.utcnow)
