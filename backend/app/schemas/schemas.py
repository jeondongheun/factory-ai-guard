from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class SensorData(BaseModel):
    timestamp: datetime
    accelerometer1: float
    accelerometer2: float
    current: float
    pressure: float
    temperature: float
    thermocouple: float
    voltage: float
    flow_rate: float

class DetectionRequest(BaseModel):
    sensor_data: List[List[float]]  # (60, 8) 윈도우

class DetectionResponse(BaseModel):
    anomaly_detected: bool
    probability: float
    severity: str
    sensor_reading_id: Optional[int]
    # RAAD-LLM 확장 필드 (Optional - 기존 클라이언트 호환)
    mode: Optional[str] = None             # 운영 모드 (high/mid/low_flow)
    z_score: Optional[float] = None        # 1순위 센서 z-score
    primary_sensor: Optional[str] = None   # 탐지에 사용된 1순위 센서
    fault_type: Optional[str] = None       # 추정 고장 유형

class DiagnosisResponse(BaseModel):
    detection_id: int
    probable_cause: str
    recommendation: str

class ThresholdUpdate(BaseModel):
    sensor_name: str
    warning_value: float
    critical_value: float

class ModelMetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    recorded_at: datetime

class StatsResponse(BaseModel):
    total_readings: int
    total_anomalies: int
    anomaly_rate: float
    avg_probability: float
