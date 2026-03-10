import anthropic
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../../../../.env'))

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
            prompt = f"""You are an expert in smart factory anomaly detection.
Sensor Statistics: {sensor_stats}
Detection Result:
- Anomaly Probability: {probability:.1%}
- Severity: {severity}

Domain Knowledge:
- Normal temperature: 20-30°C (warning: >30°C, critical: >35°C)
- Normal vibration: <0.5g (warning: >0.5g, critical: >1.0g)
- Normal flow rate: 8-12 L/min (warning: <8, critical: <6)
- Normal pressure: 1.5-2.5 Bar (warning: >2.5, critical: >3.0)
- Normal current: 8-12A (warning: >12A, critical: >15A)

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