-- 센서 데이터
CREATE TABLE IF NOT EXISTS sensor_readings (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    accelerometer1 FLOAT,
    accelerometer2 FLOAT,
    current FLOAT,
    pressure FLOAT,
    temperature FLOAT,
    thermocouple FLOAT,
    voltage FLOAT,
    flow_rate FLOAT
);

-- 탐지 결과
CREATE TABLE IF NOT EXISTS detection_results (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    anomaly_detected BOOLEAN,
    probability FLOAT,
    severity VARCHAR(10),
    sensor_reading_id INTEGER REFERENCES sensor_readings(id)
);

-- LLM 진단
CREATE TABLE IF NOT EXISTS llm_diagnoses (
    id SERIAL PRIMARY KEY,
    detection_id INTEGER REFERENCES detection_results(id),
    probable_cause TEXT,
    recommendation TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 임계값 설정
CREATE TABLE IF NOT EXISTS threshold_settings (
    id SERIAL PRIMARY KEY,
    sensor_name VARCHAR(50) UNIQUE,
    warning_value FLOAT,
    critical_value FLOAT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 모델 성능 지표
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1 FLOAT,
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

-- 기본 임계값 삽입
INSERT INTO threshold_settings (sensor_name, warning_value, critical_value) VALUES
    ('temperature', 30.0, 35.0),
    ('accelerometer1', 0.5, 1.0),
    ('accelerometer2', 0.5, 1.0),
    ('pressure', 2.5, 3.0),
    ('flow_rate', 8.0, 6.0),
    ('current', 12.0, 15.0),
    ('voltage', 220.0, 210.0)
ON CONFLICT (sensor_name) DO NOTHING;
