"""
AI 챗봇 라우터
==============
사용자가 자연어로 질문하면:
  1) ChromaDB RAG 검색 → 관련 도메인 지식 검색
  2) DB에서 최근 탐지 결과 조회 (컨텍스트 보강)
  3) Claude API (claude-haiku-4-5) 호출 → 한국어 답변
  4) 대화 이력 유지 (세션별 메모리)
"""

from fastapi import APIRouter, Request, HTTPException
from app.models.database import AsyncSessionLocal, DetectionResult, SensorReading
from sqlalchemy import select, desc
import anthropic
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# ── 대화 세션 메모리 (서버 재시작 시 초기화) ──────────────────────
# 구조: { session_id: [ {"role": "user"/"assistant", "content": "..."}, ... ] }
_sessions: dict[str, list[dict]] = {}
_MAX_HISTORY = 20   # 최대 메시지 수 (초과 시 오래된 것 제거)

# ── 논문 Figure 3 형식의 RAAD-LLM 프롬프트 빌더 ─────────────────────
# 논문: RAAD-LLM: Adaptive Anomaly Detection Using LLMs and RAG Integration
# Figure 3: INSTRUCTIONS / CONTEXT / DATA / RAG 4-섹션 구조
# Figure 4: "* High deviation is present for X." 형식의 출력

# 임계값 참조 (논문 Section 4.1.3 기반 SKAB 적용)
_THRESHOLDS = {
    "high_flow": {
        "Accelerometer1RMS":   2.56,
        "Volume_Flow_RateRMS": 1.71,
    },
    "mid_flow": {
        "Volume_Flow_RateRMS": 0.75,
    },
    "low_flow": {
        "Volume_Flow_RateRMS": 0.45,
    },
}

_MODE_KR = {
    "high_flow": "고유량(>100 L/min)",
    "mid_flow":  "중유량(50~100 L/min)",
    "low_flow":  "저유량(<50 L/min)",
}

_SENSOR_KR = {
    "Accelerometer1RMS":   "가속도계1(진동)",
    "Volume_Flow_RateRMS": "유량",
    "Temperature":         "온도",
}


def build_raad_prompt(
    mode: str,
    sensor_zscores: dict,          # {"센서명": z_score_값}
    temp_trend: dict | None,       # {"consec_dec": int, "ma_slope": float} or None
    rag_docs: list[str],
    recent_detections: list[dict],
) -> str:
    """
    논문 Figure 3 형식의 RAAD-LLM 프롬프트 생성

    구조:
      INSTRUCTIONS: 역할 + 센서별 질문 (yes/no)
      CONTEXT:      RAG 도메인 지식 (캐시된 정보)
      DATA:         각 센서 z-score 실제값
      RAG:          임계값 대비 비교 결과 (greater than / less than / equal to)
    """
    thresholds = _THRESHOLDS.get(mode, {})

    # ── 센서 목록 결정 (모드별) ───────────────────────────────────
    sensors = list(sensor_zscores.keys())
    has_temp = temp_trend is not None

    # ── INSTRUCTIONS 섹션 ─────────────────────────────────────────
    questions = []
    for s in sensors:
        kr = _SENSOR_KR.get(s, s)
        questions.append(f"* Is high deviation present for {kr}?")
    if has_temp:
        questions.append("* Is high deviation present for 온도(Temperature trend)?")

    instructions = (
        "INSTRUCTIONS: You are a helpful assistant that can use these rules to answer queries. "
        f"The following sensor data was collected over the last 60 seconds from a SKAB water pump "
        f"operating in {_MODE_KR.get(mode, mode)} mode and represents current process conditions. "
        "Strictly based on the CONTEXT and RAG information provided below, please answer the "
        "following questions. Do not modify, interpret, or apply logic beyond these instructions.\n"
        + "\n".join(questions)
        + "\nFor each question, avoid explaining. Just print only the output and nothing else."
    )

    # ── CONTEXT 섹션 (RAG 도메인 지식) ───────────────────────────
    context_lines = []
    if rag_docs:
        for doc in rag_docs[:3]:
            context_lines.append(doc[:250])
    if recent_detections:
        context_lines.append("【최근 탐지 이력】")
        for r in recent_detections[:3]:
            status = "이상" if r["anomaly_detected"] else "정상"
            context_lines.append(
                f"- {r['timestamp']} | {status} | 확률:{r['probability']:.1%}"
            )
    context_body = "\n".join(context_lines) if context_lines else "<no cached info>"

    context_section = f"CONTEXT: {context_body}"

    # ── DATA 섹션 (z-score 실제값) ────────────────────────────────
    data_parts = []
    for s, z in sensor_zscores.items():
        kr = _SENSOR_KR.get(s, s)
        data_parts.append(f"{kr} has a z-score of {z:.3f}.")
    if has_temp:
        consec = temp_trend.get("consec_dec", 0)
        slope  = temp_trend.get("ma_slope", 0.0)
        data_parts.append(
            f"온도(Temperature) exhibits a consecutive decrease trend "
            f"over {consec} data points (MA slope = {slope:.3f}°C/s)."
        )

    data_section = "DATA: " + " ".join(data_parts)

    # ── RAG 섹션 (임계값 비교) ────────────────────────────────────
    rag_parts = []
    for s, z in sensor_zscores.items():
        kr    = _SENSOR_KR.get(s, s)
        thr   = thresholds.get(s)
        if thr is None:
            continue
        if z > thr:
            comparison = f"greater than"
        elif z < thr:
            comparison = f"less than"
        else:
            comparison = "equal to"
        rag_parts.append(
            f"The z-score for {kr} is {comparison} acceptable process variable "
            f"conditions (threshold z={thr})."
        )
    if has_temp:
        consec = temp_trend.get("consec_dec", 0)
        if consec >= 10:
            comparison = "greater than"
        elif consec >= 5:
            comparison = "greater than"
        else:
            comparison = "less than"
        rag_parts.append(
            f"The temperature trend is {comparison} acceptable process variable "
            f"conditions (Warning: consec_decrease≥5, Critical: consec_decrease≥10)."
        )

    rag_section = "RAG: " + " ".join(rag_parts) if rag_parts else "RAG: No comparison data available."

    return "\n\n".join([instructions, context_section, data_section, rag_section])


# ── 일반 대화용 시스템 프롬프트 (Figure 3 스타일 적용) ────────────────
_SYSTEM_PROMPT = """You are 'FactoryGuard-AI', an expert assistant for SKAB water pump anomaly detection.

INSTRUCTIONS: Answer queries strictly based on RAAD-LLM pipeline results and domain knowledge below.
Do not modify, interpret, or apply logic beyond these instructions.

CONTEXT (Domain Knowledge):
- Operating modes: 고유량(flow>100 L/min), 중유량(50~100), 저유량(<50)
- Detection strategy: 고유량 = vibration(accel) + flow_rate / 중유량·저유량 = flow_rate only / Temperature = SPC MAMR trend
- Thresholds: 고유량 accel z>2.56 (1st), 고유량 flow z>1.71 (2nd), 중유량 z>0.75, 저유량 z>0.45
- Fault types: valve1(유량감소+온도하락), valve2(압력변동), rotor_imbalance(진동↑+유량↓), cavitation(고주파진동)
- Temperature: Warning if consecutive decrease ≥5 points, Critical if ≥10 points

RULES:
- Always respond in Korean (한국어)
- Quote actual z-score values and thresholds when available
- If anomaly detected: state sensor → z-score → threshold → probable cause → action
- If no anomaly: confirm normal operation
- Admit uncertainty honestly
"""


def _get_client() -> anthropic.Anthropic | None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def _trim_history(history: list[dict]) -> list[dict]:
    """대화 이력이 너무 길면 앞부분 제거 (최신 유지)"""
    if len(history) > _MAX_HISTORY:
        return history[-_MAX_HISTORY:]
    return history


async def _get_recent_detections(n: int = 5) -> list[dict]:
    """DB에서 최근 탐지 결과 n개 조회"""
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(DetectionResult)
                .order_by(desc(DetectionResult.timestamp))
                .limit(n)
            )
            rows = result.scalars().all()
            return [
                {
                    "timestamp":        r.timestamp.strftime("%Y-%m-%d %H:%M:%S") if r.timestamp else "N/A",
                    "anomaly_detected": r.anomaly_detected,
                    "probability":      round(r.probability or 0, 3),
                    "severity":         r.severity or "Normal",
                }
                for r in rows
            ]
    except Exception:
        return []


async def _get_latest_sensor() -> dict | None:
    """DB에서 최신 센서 값 1건 조회"""
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(SensorReading)
                .order_by(desc(SensorReading.timestamp))
                .limit(1)
            )
            row = result.scalars().first()
            if not row:
                return None
            return {
                "timestamp":     row.timestamp.strftime("%Y-%m-%d %H:%M:%S") if row.timestamp else "N/A",
                "temperature":   row.temperature,
                "accelerometer1":row.accelerometer1,
                "flow_rate":     row.flow_rate,
                "pressure":      row.pressure,
                "voltage":       row.voltage,
            }
    except Exception:
        return None


async def _get_anomaly_summary(n: int = 100) -> dict:
    """최근 n개 탐지 결과 통계"""
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(DetectionResult)
                .order_by(desc(DetectionResult.timestamp))
                .limit(n)
            )
            rows = result.scalars().all()
            if not rows:
                return {"total": 0, "anomaly": 0, "normal": 0, "anomaly_list": []}
            anomaly_rows = [r for r in rows if r.anomaly_detected]
            return {
                "total":   len(rows),
                "anomaly": len(anomaly_rows),
                "normal":  len(rows) - len(anomaly_rows),
                "anomaly_list": [
                    {
                        "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "probability": round(r.probability or 0, 3),
                        "severity": r.severity or "Normal",
                    }
                    for r in anomaly_rows[:5]
                ],
                "first_ts": rows[-1].timestamp.strftime("%Y-%m-%d %H:%M") if rows else "N/A",
                "last_ts":  rows[0].timestamp.strftime("%Y-%m-%d %H:%M")  if rows else "N/A",
            }
    except Exception:
        return {"total": 0, "anomaly": 0, "normal": 0, "anomaly_list": []}


def _rag_search(query: str, n: int = 3) -> list[str]:
    """ChromaDB RAG 검색 (없으면 빈 리스트)"""
    try:
        import sys, os
        rag_path = os.path.join(os.path.dirname(__file__), '../../ml/rag')
        if rag_path not in sys.path:
            sys.path.insert(0, rag_path)
        from chroma_embed import retrieve_domain_knowledge
        docs = retrieve_domain_knowledge(query=query, n_results=n)
        return [d["document"] for d in docs]
    except Exception:
        return []


def _build_context(rag_docs: list[str], recent: list[dict]) -> str:
    """RAG 문서 + 최근 탐지 결과 → 컨텍스트 문자열 (일반 대화용)"""
    parts = []

    if rag_docs:
        parts.append("【관련 도메인 지식】")
        for i, doc in enumerate(rag_docs, 1):
            parts.append(f"{i}. {doc[:200]}")

    if recent:
        parts.append("\n【최근 탐지 이력】")
        for r in recent:
            status = "🔴 이상" if r["anomaly_detected"] else "🟢 정상"
            parts.append(
                f"- {r['timestamp']} | {status} | 확률:{r['probability']:.1%} | 심각도:{r['severity']}"
            )

    return "\n".join(parts)


@router.post("/message")
async def chat_message(request: Request, body: dict):
    """
    챗봇 메시지 처리 (논문 Figure 3 형식 RAAD-LLM 프롬프트 지원)

    Request body:
      message:      str   - 사용자 질문
      session_id:   str   - 대화 세션 ID (없으면 신규 생성)
      sensor_data:  dict  - (선택) 실시간 탐지 결과 {mode, z_score, primary_sensor, ...}
                             있으면 Figure 3 형식 구조화 프롬프트 사용

    Response:
      reply:        str   - AI 답변
      session_id:   str   - 세션 ID
      rag_used:     bool  - RAG 검색 사용 여부
      prompt_mode:  str   - "raad_structured" | "conversational"
    """
    message     = body.get("message", "").strip()
    session_id  = body.get("session_id") or str(uuid.uuid4())[:8]
    sensor_data = body.get("sensor_data")   # 실시간 탐지 결과 (선택)

    if not message:
        raise HTTPException(status_code=400, detail="메시지가 비어있습니다")

    # ── 1) RAG 검색 ──────────────────────────────────────────────
    rag_docs = _rag_search(message, n=3)

    # ── 2) 최근 탐지 결과 조회 ───────────────────────────────────
    recent_detections = await _get_recent_detections(n=5)

    # ── 3) 대화 이력 관리 ────────────────────────────────────────
    if session_id not in _sessions:
        _sessions[session_id] = []
    history = _sessions[session_id]

    # ── 4) 프롬프트 모드 결정 ─────────────────────────────────────
    # sensor_data가 있으면 논문 Figure 3 형식 구조화 프롬프트 사용
    prompt_mode = "conversational"
    user_content = message

    if sensor_data and isinstance(sensor_data, dict):
        mode = sensor_data.get("mode", "high_flow")
        # z-score 딕셔너리 구성
        sensor_zscores = {}
        pipeline = sensor_data.get("pipeline", {})
        spc = pipeline.get("spc", {}) if pipeline else {}
        primary = spc.get("primary_result") if spc else None
        secondary = spc.get("secondary_result") if spc else None
        if primary and primary.get("sensor") and primary.get("z_score") is not None:
            sensor_zscores[primary["sensor"]] = round(float(primary["z_score"]), 3)
        if secondary and secondary.get("sensor") and secondary.get("z_score") is not None:
            sensor_zscores[secondary["sensor"]] = round(float(secondary["z_score"]), 3)

        # 온도 트렌드 정보
        temp_trend = None
        temp_res = spc.get("temp_result") if spc else None
        if temp_res and temp_res.get("is_anomaly"):
            temp_trend = {
                "consec_dec": temp_res.get("consec_dec", 0),
                "ma_slope":   temp_res.get("ma_slope", 0.0),
            }

        if sensor_zscores:
            # 논문 Figure 3 형식 프롬프트 생성
            structured_prompt = build_raad_prompt(
                mode=mode,
                sensor_zscores=sensor_zscores,
                temp_trend=temp_trend,
                rag_docs=rag_docs,
                recent_detections=recent_detections,
            )
            user_content = f"{message}\n\n{structured_prompt}"
            prompt_mode = "raad_structured"
    else:
        # 일반 대화: 기존 컨텍스트 방식
        context = _build_context(rag_docs, recent_detections)
        if context:
            user_content = f"{message}\n\n{context}"

    history.append({"role": "user", "content": user_content})
    history = _trim_history(history)

    # ── 5) Claude API 호출 ───────────────────────────────────────
    client = _get_client()

    if client is None:
        reply = await _smart_reply(message, recent_detections)
    else:
        try:
            response = client.messages.create(
                model      = "claude-haiku-4-5-20251001",
                max_tokens = 1024,
                system     = _SYSTEM_PROMPT,
                messages   = history,
            )
            reply = response.content[0].text
        except Exception as e:
            fallback = await _smart_reply(message, recent_detections)
            reply = f"⚠️ AI 응답 오류: {str(e)[:100]}\n\n{fallback}"

    # ── 6) 이력 업데이트 ─────────────────────────────────────────
    history.append({"role": "assistant", "content": reply})
    _sessions[session_id] = history

    return {
        "reply":        reply,
        "session_id":   session_id,
        "rag_used":     len(rag_docs) > 0,
        "rag_count":    len(rag_docs),
        "prompt_mode":  prompt_mode,
    }


@router.get("/demo-prompt")
async def demo_raad_prompt():
    """
    논문 Figure 3 형식의 RAAD-LLM 프롬프트 예시 반환
    시나리오: 고유량 모드 - valve1 이상 (유량감소 + 진동증가 + 온도하락)
    """
    # 시나리오 데이터 (SKAB 펌프 고유량 모드 valve1 이상)
    example_mode = "high_flow"
    example_zscores = {
        "Accelerometer1RMS":   3.50,   # 임계값 2.56 초과 → 이상
        "Volume_Flow_RateRMS": 2.85,   # 임계값 1.71 초과 → 이상
    }
    example_temp_trend = {
        "consec_dec": 8,               # 연속 8포인트 하락 → Warning
        "ma_slope":   -0.183,
    }
    example_rag = [
        "고유량 모드에서 진동(accel z>2.56)과 유량(flow z>1.71)이 동시에 이상이면 "
        "valve1 부분 폐쇄 또는 rotor_imbalance를 의심. 즉각 육안 점검 권장.",
        "온도 연속 하락(≥5포인트)은 밸브 폐쇄로 인한 유량 감소의 이차적 지표."
    ]
    example_recent = [
        {"timestamp": "2025-03-22 09:14:01", "anomaly_detected": True,  "probability": 0.921, "severity": "High"},
        {"timestamp": "2025-03-22 09:13:01", "anomaly_detected": True,  "probability": 0.884, "severity": "High"},
        {"timestamp": "2025-03-22 09:12:01", "anomaly_detected": False, "probability": 0.072, "severity": "Normal"},
    ]

    prompt = build_raad_prompt(
        mode=example_mode,
        sensor_zscores=example_zscores,
        temp_trend=example_temp_trend,
        rag_docs=example_rag,
        recent_detections=example_recent,
    )

    # 논문 Figure 4 형식의 예상 LLM 출력
    expected_output = (
        "* High deviation is present for 가속도계1(진동).\n"
        "* High deviation is present for 유량.\n"
        "* High deviation is present for 온도(Temperature trend)."
    )

    # 이진화 함수 결과 (논문 Eq.8: f(x) → {0,1})
    # 진동 + 유량 이상 동시 발생 = 상관관계 있음 → f(x) = 1 (이상)
    binarization = {
        "correlated_anomalies": ["가속도계1(진동)", "유량"],
        "f_x": 1,
        "final_prediction": "anomaly",
        "fault_type_inferred": "valve1_closure 또는 rotor_imbalance",
        "note": "논문 Eq.8: 이상 신호가 상관관계를 가지면 f(x)=1, 아니면 f(x)=0"
    }

    return {
        "scenario": {
            "description": "고유량 모드 - valve1 이상 (유량감소 + 진동증가 + 온도하락)",
            "mode": example_mode,
            "flow_rate": "118.3 L/min (정상범위: 124±2)",
            "accel1":    "0.31 g (정상범위: 0.24±0.02)",
            "temp_trend":"연속 8포인트 하락, MA slope=-0.183°C/s",
        },
        "prompt_figure3": prompt,
        "expected_output_figure4": expected_output,
        "binarization_eq8": binarization,
    }


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """대화 세션 초기화"""
    _sessions.pop(session_id, None)
    return {"cleared": True, "session_id": session_id}


@router.get("/sessions")
async def list_sessions():
    """활성 세션 수 조회 (디버그용)"""
    return {"active_sessions": len(_sessions)}


# ── 스마트 답변 (API 키 없을 때 / 베타 버전 동등 기능) ────────────────

async def _smart_reply(message: str, recent: list[dict]) -> str:
    """
    베타 interactive_chatbot.py의 기능을 웹 버전으로 구현.
    DB 실데이터 기반 키워드 매칭 응답.
    """
    msg = message.lower()

    # ── 현재 상태 / 상태 조회 ───────────────────────────────────
    if any(k in msg for k in ["현재", "지금", "상태", "어때", "current", "status"]):
        sensor = await _get_latest_sensor()
        recent_det = recent[:1]
        latest = recent_det[0] if recent_det else None

        status_str = "🔴 **이상 감지**" if (latest and latest["anomaly_detected"]) else "🟢 **정상 운전**"
        prob_str   = f"{latest['probability']:.1%}" if latest else "N/A"
        time_str   = latest["timestamp"] if latest else "N/A"

        reply = f"📊 **최근 시스템 상태** ({time_str})\n\n상태: {status_str}\n이상 확률: {prob_str}\n"

        if sensor:
            reply += (
                f"\n**센서 값:**\n"
                f"- 온도: {sensor['temperature']:.1f}°C\n"
                f"- 진동: {sensor['accelerometer1']:.3f} g\n"
                f"- 유량: {sensor['flow_rate']:.1f} L/min\n"
                f"- 압력: {sensor['pressure']:.2f} Bar\n"
                f"- 전압: {sensor['voltage']:.1f} V"
            )
        return reply

    # ── 이상/불량 시간 조회 ──────────────────────────────────────
    if any(k in msg for k in ["불량", "이상 시간", "언제", "몇 번", "몇번", "이상 발생", "anomaly"]):
        summary = await _get_anomaly_summary(n=200)
        if summary["anomaly"] == 0:
            return "✅ 최근 측정 기간 동안 **이상이 감지되지 않았습니다.** 펌프가 정상 운전 중입니다."

        reply = (
            f"⚠️ 최근 **{summary['total']}개 윈도우** 중 "
            f"**{summary['anomaly']}건의 이상**이 탐지되었습니다.\n\n"
            f"측정 기간: {summary['first_ts']} ~ {summary['last_ts']}\n\n"
            "**최근 이상 발생 시점:**\n"
        )
        for item in summary["anomaly_list"]:
            sev = item["severity"]
            reply += f"- {item['timestamp']} | 확률: {item['probability']:.1%} | 심각도: {sev}\n"
        reply += "\n가능한 원인: 밸브 부분 폐쇄, 임펠러 불균형, 베어링 마모"
        return reply

    # ── 통계 / 요약 ──────────────────────────────────────────────
    if any(k in msg for k in ["통계", "요약", "전체", "총", "summary", "statistics"]):
        summary = await _get_anomaly_summary(n=200)
        if summary["total"] == 0:
            return "아직 탐지 데이터가 없습니다. 센서 데이터를 업로드하거나 실시간 모니터링을 시작하세요."
        ratio = summary["anomaly"] / summary["total"] * 100
        return (
            f"📊 **전체 탐지 통계**\n\n"
            f"- 분석 기간: {summary['first_ts']} ~ {summary['last_ts']}\n"
            f"- 총 측정: {summary['total']}회\n"
            f"- 정상: {summary['normal']}회 ({100 - ratio:.1f}%)\n"
            f"- 이상: {summary['anomaly']}회 ({ratio:.1f}%)\n"
            f"\n최근 탐지 이력 페이지에서 상세 내역을 확인하세요."
        )

    # ── 온도 ────────────────────────────────────────────────────
    if any(k in msg for k in ["온도", "temperature", "열"]):
        sensor = await _get_latest_sensor()
        temp_str = f"{sensor['temperature']:.1f}°C" if sensor else "데이터 없음"
        return (
            f"🌡️ **온도 이상 탐지** (현재: {temp_str})\n\n"
            "온도는 SPC MAMR 트렌드로 판별합니다:\n"
            "- **연속 5포인트 하락**: Warning (밸브 부분 폐쇄 의심)\n"
            "- **연속 10포인트 하락**: Critical (즉시 점검 필요)\n\n"
            "밸브 폐쇄 → 유량 감소 → 펌프 내 열 교환 감소 → 온도 하락 패턴입니다."
        )

    # ── 진동 ────────────────────────────────────────────────────
    if any(k in msg for k in ["진동", "vibration", "accelerometer", "accel"]):
        sensor = await _get_latest_sensor()
        accel_str = f"{sensor['accelerometer1']:.3f} g" if sensor else "데이터 없음"
        return (
            f"📳 **진동 이상 탐지** (현재: {accel_str})\n\n"
            "고유량 모드에서만 진동을 탐지합니다 (SNR 확보):\n"
            "- **DFT A_max z-score > 2.56**: 이상 판정\n"
            "- 주요 원인: 로터 불균형, 베어링 마모, 공동현상(cavitation)\n\n"
            "중유량/저유량에서는 진동 탐지를 비활성화합니다."
        )

    # ── 유량 ────────────────────────────────────────────────────
    if any(k in msg for k in ["유량", "flow", "밸브", "valve"]):
        sensor = await _get_latest_sensor()
        flow_str = f"{sensor['flow_rate']:.1f} L/min" if sensor else "데이터 없음"
        return (
            f"💧 **유량 이상 임계값** (현재: {flow_str})\n\n"
            "모드별 z-score 임계값:\n"
            "- 고유량(>100 L/min): z > **1.71**\n"
            "- 중유량(50~100): z > **0.75**\n"
            "- 저유량(<50): z > **0.45**\n\n"
            "유량 감소 원인: 밸브 이상, 배관 막힘, 임펠러 손상"
        )

    # ── 고장 유형 ─────────────────────────────────────────────
    if any(k in msg for k in ["고장", "원인", "유형", "종류", "fault", "cause"]):
        return (
            "🔧 **탐지 가능한 고장 유형:**\n\n"
            "1. **valve1 부분 폐쇄**: 유량 감소 + 온도 하락 동시 발생\n"
            "2. **valve2 개도 변동**: 압력 변동 + 유량 불안정\n"
            "3. **Rotor Imbalance**: 고주파 진동 급증 + 유량 감소\n"
            "4. **Cavitation**: 광대역 진동 + 특정 주파수 성분 소실\n"
            "5. **Bearing Wear**: 저주파 진동 + 온도 서서히 상승"
        )

    # ── RAAD-LLM / SPC / DFT 설명 ────────────────────────────
    if any(k in msg for k in ["raad", "spc", "dft", "알고리즘", "탐지 방법", "어떻게"]):
        return (
            "🧠 **RAAD-LLM 탐지 파이프라인:**\n\n"
            "**① SPC MAMR** → 신호 정제 + UCL/LCL 계산\n"
            "**② DFT** → 주파수 분석, A_max·f_max 추출\n"
            "**③ Z-score** → 모드별 baseline 대비 이상 점수\n"
            "**④ Claude LLM** → Figure 3 구조화 프롬프트 → Figure 4 출력\n"
            "**⑤ Eq.8 이진화** → f(x) ∈ {0,1}\n"
            "**⑥ Adaptability** → f(x)=0이면 EMA(α=0.02)로 baseline 업데이트"
        )

    # ── 도움말 / 기본 ────────────────────────────────────────
    return (
        "🤖 **FactoryGuard-AI 챗봇**입니다.\n\n"
        "다음과 같이 질문해보세요:\n"
        "- '현재 상태 알려줘' → 실시간 센서값 + 탐지 상태\n"
        "- '이상 언제 났어?' → 이상 발생 시간 목록\n"
        "- '전체 통계 요약해줘' → 탐지 기간 통계\n"
        "- '온도 이상 원인이 뭐야?' → 도메인 지식 설명\n"
        "- '고유량 모드 진동 기준' → 임계값 설명\n"
        "- '탐지 가능한 고장 유형' → 고장 유형 목록\n\n"
        "*(ANTHROPIC_API_KEY 설정 시 Claude AI가 더 정확하게 답변합니다)*"
    )
