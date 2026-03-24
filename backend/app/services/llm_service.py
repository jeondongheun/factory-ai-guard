import anthropic
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../../../../.env'))

# RAG 모듈 경로 등록
_RAG_DIR = Path(__file__).parent.parent.parent / "ml" / "rag"
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))


def _infer_mode_from_stats(sensor_stats: dict) -> str:
    """sensor_stats의 유량값으로 운영 모드 추정"""
    flow = sensor_stats.get("flow_rate", sensor_stats.get("Volume_Flow_RateRMS", None))
    if flow is None:
        return "high_flow"
    flow = float(flow)
    if flow > 100:
        return "high_flow"
    elif flow > 50:
        return "mid_flow"
    return "low_flow"


def _build_rag_domain_section(
    sensor_stats: dict,
    severity: str,
    probability: float,
) -> str:
    """
    ChromaDB RAG 검색으로 도메인 지식 섹션 생성.
    실패 시 하드코딩 기본값으로 fallback.
    """
    try:
        from chroma_embed import retrieve_for_llm

        mode = _infer_mode_from_stats(sensor_stats)

        # severity → z_score 근사 (탐지 결과가 z_score를 직접 전달하지 않으므로)
        z_approx = {"High": 3.5, "Medium": 2.0, "Low": 1.2, "Normal": 0.3}.get(severity, 1.5)

        # 1순위 센서 추정 (유량 이상이면 Flow, 아니면 Accel)
        accel = float(sensor_stats.get("accelerometer1", sensor_stats.get("Accelerometer1RMS", 0)))
        flow  = float(sensor_stats.get("flow_rate", sensor_stats.get("Volume_Flow_RateRMS", 100)))
        primary_sensor = "Accelerometer1RMS" if (accel > 0.3 and mode == "high_flow") else "Volume_Flow_RateRMS"

        temp  = float(sensor_stats.get("temperature", sensor_stats.get("Temperature", 25)))
        temp_anomaly = temp < 15 or temp > 40

        rag = retrieve_for_llm(
            mode           = mode,
            z_score        = z_approx,
            primary_sensor = primary_sensor,
            fault_type     = "rotor_imbalance_suspected" if "Accel" in primary_sensor else "valve_anomaly_low",
            temp_anomaly   = temp_anomaly,
            n_results      = 3,
        )

        lines = []
        for key in ("strategy", "thresholds", "fault_info", "temp_info"):
            for doc in rag.get(key, [])[:2]:
                lines.append(f"- {doc['document']}")

        if lines:
            return "Domain Knowledge (from RAG):\n" + "\n".join(lines)

    except Exception:
        pass

    # Fallback: 하드코딩 기본 도메인 지식
    return (
        "Domain Knowledge:\n"
        "- Normal temperature: 20-30°C (warning: >30°C, critical: >35°C)\n"
        "- Normal vibration: <0.5g (warning: >0.5g, critical: >1.0g)\n"
        "- Normal flow rate: 8-12 L/min (warning: <8, critical: <6)\n"
        "- Normal pressure: 1.5-2.5 Bar (warning: >2.5, critical: >3.0)\n"
        "- Normal current: 8-12A (warning: >12A, critical: >15A)"
    )


class LLMService:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("⚠️  ANTHROPIC_API_KEY 없음 - Mock 모드로 실행")
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=api_key)

    async def diagnose(self, probability: float, severity: str, sensor_stats: dict) -> dict:
        if self.client is None:
            return self._mock_diagnose(probability, severity)
        try:
            # RAG로 도메인 지식 검색 (ChromaDB 없으면 자동 fallback)
            domain_section = _build_rag_domain_section(sensor_stats, severity, probability)

            prompt = f"""You are an expert in smart factory anomaly detection.
Sensor Statistics: {sensor_stats}
Detection Result:
- Anomaly Probability: {probability:.1%}
- Severity: {severity}

{domain_section}

Analyze the anomaly and respond in JSON:
{{
  "probable_cause": "한국어로 원인 설명 (2문장 이내)",
  "recommendation": "한국어로 권장 조치 (2문장 이내)"
}}"""
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )
            import json, re
            response = message.content[0].text
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._mock_diagnose(probability, severity)
        except Exception as e:
            print(f"❌ LLM 오류: {e}")
            return self._mock_diagnose(probability, severity)

    def _mock_diagnose(self, probability: float, severity: str) -> dict:
        causes = {
            "High":   "베어링 마모 또는 임펠러 손상이 의심됩니다. 다수 센서에서 임계값 초과가 감지되었습니다.",
            "Medium": "센서 값이 경고 범위에 진입했습니다. 온도 또는 진동 수치가 상승 중입니다.",
            "Low":    "경미한 이상 징후가 감지되었습니다. 지속적인 모니터링이 필요합니다.",
            "Normal": "정상 범위 내에서 운전 중입니다."
        }
        recommendations = {
            "High":   "즉시 운전을 중단하고 전문가 점검을 받으세요. 베어링 및 윤활유 상태를 확인하세요.",
            "Medium": "운전을 유지하되 30분 내 점검을 권장합니다. 온도 및 진동 추이를 주시하세요.",
            "Low":    "정기 점검 일정에 포함하여 확인하세요.",
            "Normal": "현재 상태를 유지하며 정기 점검을 계속하세요."
        }
        return {
            "probable_cause": causes.get(severity, causes["Normal"]),
            "recommendation": recommendations.get(severity, recommendations["Normal"])
        }