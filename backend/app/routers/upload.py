from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from app.models.database import AsyncSessionLocal, DetectionResult, SensorReading
import pandas as pd
import numpy as np
import io
import uuid
import os

router = APIRouter()

# 배치 분석 결과 임시 캐시 (job_id → 요약)
job_results = {}

# SKAB 8채널 순서: [accel1, accel2, current, pressure, temp, thermo, voltage, flow]
_ORDERED_COLS = [
    'Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure',
    'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS',
]
_COL_ALIASES = {
    'Accelerometer1RMS':   ['Accelerometer1RMS', 'accelerometer1'],
    'Accelerometer2RMS':   ['Accelerometer2RMS', 'accelerometer2'],
    'Current':             ['Current', 'current'],
    'Pressure':            ['Pressure', 'pressure'],
    'Temperature':         ['Temperature', 'temperature'],
    'Thermocouple':        ['Thermocouple', 'thermocouple'],
    'Voltage':             ['Voltage', 'voltage'],
    'Volume Flow RateRMS': ['Volume Flow RateRMS', 'Volume_Flow_RateRMS', 'flow_rate'],
}


def _resolve_columns(df: pd.DataFrame) -> list[str]:
    """CSV 컬럼명 → 실제 df 컬럼명 매핑 (8채널 순서 유지)"""
    resolved = []
    for canonical in _ORDERED_COLS:
        for alias in _COL_ALIASES[canonical]:
            if alias in df.columns:
                resolved.append(alias)
                break
        else:
            resolved.append(None)   # 해당 채널 없음
    return resolved


@router.post("/csv")
async def upload_csv(request: Request, file: UploadFile = File(...)):
    """CSV 파일 배치 분석 + DB 저장 (통계/이력 페이지에 반영)"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="CSV 파일만 업로드 가능합니다")

    ml_service = request.app.state.ml_service

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')), sep=None, engine='python')
    df = df.fillna(method='ffill').fillna(0)

    col_map = _resolve_columns(df)
    available_count = sum(1 for c in col_map if c is not None)
    if available_count < 4:
        raise HTTPException(
            status_code=400,
            detail=f"센서 컬럼 부족 (최소 4개 필요). 발견: {list(df.columns)}"
        )

    # 8채널 배열 구성 (없는 채널은 0)
    n = len(df)
    features = np.zeros((n, 8))
    for ch_idx, col_name in enumerate(col_map):
        if col_name is not None:
            features[:, ch_idx] = df[col_name].values

    window_size = 60
    step        = 10
    results     = []

    # ANTHROPIC_API_KEY가 있으면 완전한 RAAD-LLM 파이프라인 사용 (LLM in the loop)
    # 없으면 SPC 전용 모드 (빠름)
    api_key     = os.getenv("ANTHROPIC_API_KEY")
    use_llm     = bool(api_key)
    llm_updates = 0   # 적응형 업데이트 횟수

    async with AsyncSessionLocal() as db:
        for i in range(0, n - window_size, step):
            window_arr = features[i:i + window_size]   # (60, 8)

            # 논문 Figure 2: LLM in the loop (API 키 있을 때)
            if use_llm:
                result = ml_service.detect_with_llm(window_arr.tolist(), api_key=api_key)
                if result.get("adaptability_updated"):
                    llm_updates += 1
            else:
                result = ml_service.detect(window_arr.tolist())

            # 윈도우 평균으로 SensorReading 저장
            mean_v = window_arr.mean(axis=0)
            sensor_row = SensorReading(
                accelerometer1 = float(mean_v[0]),
                accelerometer2 = float(mean_v[1]),
                current        = float(mean_v[2]),
                pressure       = float(mean_v[3]),
                temperature    = float(mean_v[4]),
                thermocouple   = float(mean_v[5]),
                voltage        = float(mean_v[6]),
                flow_rate      = float(mean_v[7]),
            )
            db.add(sensor_row)
            await db.flush()   # id 확보

            det_row = DetectionResult(
                anomaly_detected  = bool(result["anomaly_detected"]),
                probability       = float(result["probability"]),
                severity          = str(result["severity"]),
                sensor_reading_id = sensor_row.id,
            )
            db.add(det_row)

            row_entry = {
                "index":            i,
                "anomaly_detected": result["anomaly_detected"],
                "probability":      result["probability"],
                "severity":         result["severity"],
                "mode":             result.get("mode"),
                "z_score":          result.get("z_score"),
                "fault_type":       result.get("fault_type"),
            }
            # LLM 파이프라인 추가 정보 (API 키 있을 때만)
            if use_llm:
                row_entry["llm_prediction"] = result.get("llm_prediction")
                row_entry["llm_override"]   = result.get("llm_override", False)
            results.append(row_entry)

        await db.commit()

    # ── 요약 통계 ────────────────────────────────────────────────
    total      = len(results)
    anomalies  = sum(1 for r in results if r["anomaly_detected"])
    by_severity: dict = {}
    for r in results:
        s = r["severity"]
        by_severity[s] = by_severity.get(s, 0) + 1

    llm_overrides = sum(1 for r in results if r.get("llm_override"))

    job_id = str(uuid.uuid4())[:8]
    job_results[job_id] = {
        "filename":        file.filename,
        "total_windows":   total,
        "anomaly_count":   anomalies,
        "anomaly_rate":    round(anomalies / total, 4) if total > 0 else 0,
        "by_severity":     by_severity,
        "pipeline_mode":   "RAAD-LLM (LLM in the loop)" if use_llm else "SPC-only",
        "llm_overrides":   llm_overrides,      # LLM이 SPC 결과를 뒤집은 횟수
        "adaptability_updates": llm_updates,   # baseline 자동 업데이트 횟수
        "results":         results,
    }

    return {
        "job_id":        job_id,
        "total_windows": total,
        "anomaly_count": anomalies,
        "anomaly_rate":  round(anomalies / total, 4) if total > 0 else 0,
        "by_severity":   by_severity,
        "saved_to_db":   True,
    }


@router.get("/result/{job_id}")
async def get_result(job_id: str):
    """배치 분석 결과 조회"""
    if job_id not in job_results:
        raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다")
    return job_results[job_id]
