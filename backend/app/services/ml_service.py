"""
MLService - RAAD-LLM 기반 이상 탐지 서비스
============================================
파이프라인 (논문 Section 4.1.3):

  [고유량 모드]
    Accelerometer1RMS → SPC 2-pass 필터 → DFT → A_max z-score > 2.56  (1순위)
    Volume_Flow_Rate  → 고유량 baseline z-score > 1.71                  (2순위)
    Temperature       → MAMR 연속 하락 트렌드 탐지                      (보조)

  [중유량 모드]
    Volume_Flow_Rate  → 중유량 baseline z-score > 0.75                  (단독)

  [저유량 모드]
    Volume_Flow_Rate  → 저유량 baseline z-score > 0.45                  (단독)

  [LSTM Encoder]
    기존 LSTM 모델은 보조 신호로 유지 (baseline 미초기화 시 fallback)

센서 컬럼 순서 (sensor_window shape: (60, 8)):
  0: Accelerometer1RMS   1: Accelerometer2RMS   2: Current
  3: Pressure            4: Temperature         5: Thermocouple
  6: Voltage             7: Volume_Flow_RateRMS
"""

from __future__ import annotations

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ── RAG 모듈 경로 등록 ─────────────────────────────────────────
_RAG_DIR = Path(__file__).parent.parent.parent / "ml" / "rag"
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))

from spc_processor import SPCProcessor, get_operating_mode   # noqa: E402
from dft_processor  import VibrationPipeline                 # noqa: E402


# ─────────────────────────────────────────────────────────────────
# 센서 컬럼 인덱스
# ─────────────────────────────────────────────────────────────────
IDX_ACCEL1 = 0
IDX_TEMP   = 4
IDX_FLOW   = 7

# CSV 기반 기본 baseline (서버 시작 직후 학습 데이터 없을 때 사용)
# 출처: rag_knowledge_base_bom.csv id=1~9
_DEFAULT_BASELINES = {
    "high_flow": {
        "Accelerometer1RMS":   {"mean": 0.24,  "std": 0.02},
        "Volume_Flow_RateRMS": {"mean": 124.0, "std": 2.0 },
    },
    "mid_flow": {
        "Volume_Flow_RateRMS": {"mean": 75.0,  "std": 2.0 },
    },
    "low_flow": {
        "Volume_Flow_RateRMS": {"mean": 32.0,  "std": 1.0 },
    },
}
_DEFAULT_BASELINE_N = 300   # synthetic 포인트 수


# ─────────────────────────────────────────────────────────────────
# LSTM Encoder (기존 모델 - 보조 신호로 유지)
# ─────────────────────────────────────────────────────────────────

class TimeSeriesEncoder(nn.Module):
    """LSTM 기반 시계열 인코더 (기존 구조 유지)"""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


# ─────────────────────────────────────────────────────────────────
# MLService
# ─────────────────────────────────────────────────────────────────

class MLService:
    """
    RAAD-LLM 기반 이상 탐지 서비스

    detect() 메서드가 메인 파이프라인.
    predict()는 기존 API 호환성 유지용 래퍼.
    """

    def __init__(self):
        self.device     = "cpu"
        self.model      = None
        self.scaler     = StandardScaler()
        self.window_size = 60
        self.input_dim   = 8
        self.model_path  = (
            Path(__file__).parent.parent.parent / "ml" / "weights" / "best_model.pth"
        )
        self.is_loaded   = False
        self.model_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        # ── RAAD-LLM 파이프라인 컴포넌트 ─────────────────────────
        self.spc         = SPCProcessor(spc_passes=2)
        self.vibration   = VibrationPipeline(
            window_length=60,
            z_threshold=2.56,
            sample_rate=1.0,
            spc_passes=2,
        )
        self._pipeline_ready = False

    # ─────────────────────────────────────────────────────────────
    # 초기화
    # ─────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """서버 시작 시 1회 호출: LSTM 로드 + RAAD-LLM baseline 초기화"""

        # 1) LSTM 로드
        self._load_lstm()

        # 2) RAAD-LLM baseline 초기화 (CSV 기본값으로 시드)
        self._init_default_baselines()

        print("✅ MLService 초기화 완료 (RAAD-LLM 파이프라인 + LSTM 보조)")

    def _load_lstm(self) -> None:
        """LSTM 가중치 로드"""
        try:
            self.model = TimeSeriesEncoder(
                input_dim=self.input_dim, hidden_dim=128, num_layers=2
            ).to(self.device)

            if self.model_path.exists():
                ckpt = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(ckpt["encoder"])
                if "metrics" in ckpt:
                    self.model_metrics = ckpt["metrics"]
                print(f"  ✓ LSTM 로드: {self.model_path}")
            else:
                print("  ⚠ best_model.pth 없음 → 랜덤 가중치 (LSTM은 보조 신호만)")

            self.model.eval()
            self.is_loaded = True

        except Exception as e:
            print(f"  ✗ LSTM 로드 실패: {e} (RAAD-LLM 단독 모드로 동작)")
            self.is_loaded = False

    def _init_default_baselines(self) -> None:
        """
        CSV 기반 기본값으로 SPC/DFT baseline 초기화
        실제 운영 데이터가 누적되면 적응형 업데이트로 자동 교체됨
        """
        rng = np.random.default_rng(42)

        for mode, sensors in _DEFAULT_BASELINES.items():
            for sensor, params in sensors.items():
                data = rng.normal(params["mean"], params["std"], _DEFAULT_BASELINE_N)
                self.spc.fit_baseline(mode, sensor, data)

        # VibrationPipeline baseline (고유량 진동)
        accel_norm = rng.normal(
            _DEFAULT_BASELINES["high_flow"]["Accelerometer1RMS"]["mean"],
            _DEFAULT_BASELINES["high_flow"]["Accelerometer1RMS"]["std"],
            _DEFAULT_BASELINE_N,
        )
        self.vibration.fit_baseline(accel_norm)

        self._pipeline_ready = True
        print("  ✓ RAAD-LLM baseline 초기화 (CSV 기본값 기반)")

    # ─────────────────────────────────────────────────────────────
    # 메인 탐지 메서드
    # ─────────────────────────────────────────────────────────────

    def detect(self, sensor_window: list) -> dict:
        """
        RAAD-LLM 파이프라인 기반 이상 탐지 (메인)

        Args:
            sensor_window: (60, 8) 센서 윈도우
                           컬럼: [accel1, accel2, current, pressure,
                                  temp, thermocouple, voltage, flow_rate]

        Returns:
            dict:
              anomaly_detected  (bool)
              probability       (float 0~1, 하위 호환)
              severity          (str: Normal/Low/Medium/High/Critical)
              mode              (str: high_flow/mid_flow/low_flow)
              z_score           (float: 1순위 센서 z-score)
              primary_sensor    (str: 탐지에 사용된 1순위 센서)
              fault_type        (str: 추정 고장 유형)
              pipeline          (dict: 상세 파이프라인 결과)
        """
        window = np.array(sensor_window, dtype=float)   # (60, 8)

        # ── 센서 시계열 추출 ──────────────────────────────────────
        accel1_series = window[:, IDX_ACCEL1]
        temp_series   = window[:, IDX_TEMP]
        flow_series   = window[:, IDX_FLOW]

        # 유량: 윈도우 평균 사용 (단일 포인트보다 노이즈에 강건)
        mean_flow  = float(np.mean(flow_series))
        # 온도: 스트리밍 업데이트용 최신값 + 배치 탐지용 전체 시계열 병행
        current_temp = float(temp_series[-1])

        # ── 운영 모드 판별 (윈도우 평균 기준) ────────────────────
        mode = get_operating_mode(mean_flow)

        # ── RAAD-LLM SPC 파이프라인 ───────────────────────────────
        if self._pipeline_ready:
            spc_result = self.spc.process(
                mode=mode,
                flow_value=mean_flow,
                temp_value=current_temp,
                accel_window=accel1_series if mode == "high_flow" else None,
                update_on_normal=True,
            )
            # 온도: 배치 탐지 (윈도우 내 트렌드 즉시 포착)
            temp_batch = self.spc.detect_temperature_batch(temp_series)
            if temp_batch.is_anomaly:
                spc_result["temp_result"] = {
                    "is_anomaly":  True,
                    "trend":       temp_batch.trend_direction,
                    "consec_dec":  temp_batch.consecutive_decrease,
                    "ma_slope":    round(temp_batch.ma_slope, 4),
                    "severity":    temp_batch.severity,
                    "detail":      temp_batch.detail,
                }
                if temp_batch.severity == "critical":
                    spc_result["is_anomaly"] = True
                    if spc_result["severity"] == "normal":
                        spc_result["severity"] = "warning"
        else:
            spc_result = {
                "is_anomaly": False, "severity": "normal",
                "primary_result": None, "secondary_result": None,
                "temp_result": None,
            }

        # ── 고유량 전용: DFT 파이프라인 (스펙트럼 분석 + 고장 유형 분류) ────
        # 역할: 이상 판정은 SPC z-score가 담당, DFT는 fault_type 분류용
        # 이유: DFT baseline은 실제 펌프 데이터로 fit 해야 신뢰도 확보 가능.
        #       서버 시작 시 CSV 기본값으로만 시드된 DFT baseline은 오탐 가능성이
        #       높으므로, 실제 운영 데이터가 충분히 쌓일 때까지는 fault_type
        #       레이블 생성(LLM 프롬프트 보강)에만 사용한다.
        dft_result = None
        if mode == "high_flow":
            try:
                dft_result = self.vibration.detect(accel1_series)
                # DFT는 fault_type 및 스펙트럼 정보만 활용 (is_anomaly에 반영 안 함)
                # 추후 실제 데이터 기반 DFT baseline이 안정화되면 아래를 활성화:
                # if dft_result.is_anomaly and dft_result.z_score > 5.0:
                #     spc_result["is_anomaly"] = True
                #     ...
            except Exception as e:
                dft_result = None
                spc_result.setdefault("details", {})["dft_error"] = str(e)

        # ── LSTM 보조 신호 ────────────────────────────────────────
        lstm_prob = self._lstm_score(window)

        # ── 최종 확률 합성 ────────────────────────────────────────
        raad_prob = _severity_to_prob(spc_result["severity"])
        probability = (
            0.7 * raad_prob + 0.3 * lstm_prob
            if self.is_loaded
            else raad_prob
        )
        probability = round(float(np.clip(probability, 0.0, 1.0)), 4)

        # ── 반환값 정리 ───────────────────────────────────────────
        is_anomaly = spc_result["is_anomaly"]
        severity   = _normalize_severity(spc_result["severity"], is_anomaly)

        # 1순위 센서 z-score 추출
        z_score, primary_sensor = _extract_primary_zscore(spc_result)

        # 고장 유형 (DFT 결과 우선, 없으면 모드·센서 기반 추정)
        fault_type = _infer_fault_type(mode, dft_result, spc_result)

        return {
            # ── 기존 API 호환 필드 ──
            "anomaly_detected": is_anomaly,
            "probability":      probability,
            "severity":         severity,
            # ── 신규 확장 필드 ──────
            "mode":             mode,
            "z_score":          z_score,
            "primary_sensor":   primary_sensor,
            "fault_type":       fault_type,
            "pipeline": {
                "spc":  spc_result,
                "dft":  _dft_to_dict(dft_result),
                "lstm": {"probability": round(lstm_prob, 4)},
            },
        }

    def predict(self, sensor_window: list) -> dict:
        """기존 API 호환 래퍼 → detect() 위임"""
        return self.detect(sensor_window)

    # ─────────────────────────────────────────────────────────────
    # 완전한 RAAD-LLM 파이프라인 (LLM in the loop)
    # ─────────────────────────────────────────────────────────────

    def detect_with_llm(self, sensor_window: list, api_key: str | None = None) -> dict:
        """
        논문 Figure 2 완전 구현: SPC → 프롬프트(Figure 3) → LLM → 이진화(Eq.8) → 적응형 업데이트

        Args:
            sensor_window: (60, 8) 센서 윈도우
            api_key:        Anthropic API 키 (없으면 SPC 결과만 반환)

        Returns:
            detect() 결과 + LLM 레이어 필드:
              llm_output            (str)  Figure 4 형식 원본 출력
              llm_anomalous_sensors (list) 이상 판정된 센서 목록
              llm_prediction        (int)  f(x) ∈ {0, 1}  (Eq.8)
              llm_override          (bool) LLM이 SPC 결과를 뒤집었는지
              adaptability_updated  (bool) baseline 업데이트 여부
        """
        # ── Step 1: SPC 파이프라인 실행 ──────────────────────────
        result = self.detect(sensor_window)

        if not api_key:
            return result   # API 키 없으면 SPC 결과만

        # ── Step 2: 센서 z-score 추출 → 프롬프트 빌드 ──────────
        mode    = result["mode"]
        spc_res = result.get("pipeline", {}).get("spc", {})

        sensor_zscores = {}
        for key in ("primary_result", "secondary_result"):
            r = spc_res.get(key)
            if r and r.get("sensor") and r.get("z_score") is not None:
                sensor_zscores[r["sensor"]] = round(float(r["z_score"]), 3)

        temp_trend = None
        temp_r = spc_res.get("temp_result")
        if temp_r and temp_r.get("is_anomaly"):
            temp_trend = {
                "consec_dec": temp_r.get("consec_dec", 0),
                "ma_slope":   temp_r.get("ma_slope", 0.0),
            }

        if not sensor_zscores:
            return result   # z-score 없으면 LLM 불필요

        # ── ChromaDB RAG 검색 (논문 Figure 2 RAG 단계) ───────────
        rag_context = None
        try:
            from chroma_embed import retrieve_for_llm as _rag_retrieve
            rag_context = _rag_retrieve(
                mode           = mode,
                z_score        = max(sensor_zscores.values()),
                primary_sensor = max(sensor_zscores, key=sensor_zscores.get),
                fault_type     = result.get("fault_type", "unknown"),
                temp_anomaly   = bool(temp_trend),
                n_results      = 3,
            )
        except Exception as _rag_err:
            # ChromaDB 미구축 또는 오류 → 하드코딩 임계값으로 fallback
            rag_context = None

        prompt = _build_raad_prompt_sync(mode, sensor_zscores, temp_trend, rag_context)

        # ── Step 3: Claude Haiku API 호출 ────────────────────────
        try:
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model      = "claude-haiku-4-5-20251001",
                max_tokens = 256,
                system     = (
                    "You are an anomaly detection classifier. "
                    "Answer ONLY with the bullet-point format shown. "
                    "No explanations, no extra text."
                ),
                messages=[{"role": "user", "content": prompt}],
            )
            llm_output = resp.content[0].text.strip()
        except Exception as e:
            result["llm_error"] = str(e)[:120]
            return result

        # ── Step 4: Figure 4 출력 파싱 + Eq.8 이진화 ─────────────
        anomalous = _parse_llm_output(llm_output)
        f_x       = _binarize_eq8(mode, anomalous)

        result["llm_output"]            = llm_output
        result["llm_anomalous_sensors"] = anomalous
        result["llm_prediction"]        = f_x

        # ── Step 5: LLM 판단으로 SPC 결과 보정 ──────────────────
        llm_override = False
        if f_x == 0 and result["anomaly_detected"]:
            # LLM이 정상 판정 → SPC 이상 오버라이드 (false positive 억제)
            result["anomaly_detected"] = False
            result["severity"]         = "Normal"
            result["probability"]      = max(0.02, result["probability"] * 0.3)
            llm_override = True
        elif f_x == 1 and not result["anomaly_detected"]:
            # LLM이 이상 판정 → SPC 정상 오버라이드 (false negative 보완)
            result["anomaly_detected"] = True
            result["severity"]         = "Medium"
            result["probability"]      = min(0.98, result["probability"] + 0.4)
            llm_override = True

        result["llm_override"] = llm_override

        # ── Step 6: Adaptability — f(x)=0이면 현재 윈도우로 baseline 업데이트 ──
        # 논문 Figure 2 ⑥번: "If output indicates no anomalies, update C with Q^(p)"
        adapted = False
        if f_x == 0 and self._pipeline_ready:
            window_arr = np.array(sensor_window, dtype=float)
            adapted = self._adapt_baseline(mode, window_arr, sensor_zscores)

        result["adaptability_updated"] = adapted
        return result

    def _adapt_baseline(
        self,
        mode: str,
        window: np.ndarray,
        sensor_zscores: dict,
    ) -> bool:
        """
        논문 Adaptability Mechanism 구현 (Figure 2 ⑥번)
        "If output indicates no anomalies, update C with Q^(p)"

        실제 정상 윈도우 데이터를 직접 SPC/DFT baseline에 반영.
        합성 데이터(rng.normal) 대신 실제 센서값으로 업데이트해 정확도 유지.
        """
        updated = False

        try:
            if mode == "high_flow":
                # ── Accelerometer1RMS: 실제 윈도우 데이터로 SPC baseline 업데이트
                accel_window = window[:, IDX_ACCEL1]
                self.spc.update_baseline(mode, "Accelerometer1RMS", accel_window)

                # ── VibrationPipeline(DFT) baseline: 정상 판정 윈도우로 점진 업데이트
                try:
                    dft_check = self.vibration.detect(accel_window)
                    # detect() 내부에서 정상이면 자동 update_baseline() 호출됨
                    # 추가로 명시적 반영 (이중 업데이트 방지 위해 정상 시에만)
                    if not dft_check.is_anomaly:
                        self.vibration.update_baseline(dft_check.A_max, dft_check.f_max)
                except Exception:
                    pass

                # ── _DEFAULT_BASELINES EMA 동기화 (get_pipeline_status() 표시용)
                _DEFAULT_BASELINES["high_flow"]["Accelerometer1RMS"]["mean"] = float(
                    np.mean(accel_window)
                )

            # ── Volume_Flow_RateRMS: 모든 모드에서 실제 유량 데이터로 업데이트
            if mode in _DEFAULT_BASELINES and "Volume_Flow_RateRMS" in _DEFAULT_BASELINES[mode]:
                flow_window = window[:, IDX_FLOW]
                self.spc.update_baseline(mode, "Volume_Flow_RateRMS", flow_window)
                _DEFAULT_BASELINES[mode]["Volume_Flow_RateRMS"]["mean"] = float(
                    np.mean(flow_window)
                )

            updated = True

        except Exception:
            pass

        return updated

    # ─────────────────────────────────────────────────────────────
    # LSTM 보조 추론
    # ─────────────────────────────────────────────────────────────

    def _lstm_score(self, window: np.ndarray) -> float:
        """LSTM 인코더로 이상 확률 계산 (보조 신호)"""
        if not self.is_loaded or self.model is None:
            return self._rule_score(window)
        try:
            scaled = self.scaler.fit_transform(window)
            tensor = torch.FloatTensor(scaled).unsqueeze(0).to(self.device)
            with torch.no_grad():
                encoded = self.model(tensor)
                return float(torch.sigmoid(encoded.mean()).item())
        except Exception:
            return self._rule_score(window)

    def _rule_score(self, window: np.ndarray) -> float:
        """최소 안전망: 단순 규칙 기반 점수 (LSTM도 없을 때)"""
        latest = window[-1]
        score  = 0.0
        if latest[IDX_TEMP]  > 95:   score += 0.4
        if latest[IDX_ACCEL1] > 0.5: score += 0.3
        if latest[IDX_FLOW]  < 10:   score += 0.3
        return float(np.clip(score, 0.0, 0.99))

    # ─────────────────────────────────────────────────────────────
    # 외부 인터페이스
    # ─────────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """저장된 모델 성능 지표"""
        return self.model_metrics

    def get_pipeline_status(self) -> dict:
        """파이프라인 초기화 상태 및 baseline 정보"""
        status = {
            "pipeline_ready":   self._pipeline_ready,
            "lstm_loaded":      self.is_loaded,
            "vibration_baseline": self.vibration.baseline_info,
            "spc_baselines":    {},
        }
        for mode in ["high_flow", "mid_flow", "low_flow"]:
            sensors = (
                ["Accelerometer1RMS", "Volume_Flow_RateRMS"]
                if mode == "high_flow"
                else ["Volume_Flow_RateRMS"]
            )
            status["spc_baselines"][mode] = {
                s: self.spc.zscore_det.get_baseline_info(mode, s)
                for s in sensors
            }
        return status


# ─────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────

def _severity_to_prob(severity: str) -> float:
    """severity 문자열 → 확률 (하위 호환용)"""
    return {
        "critical": 0.92,
        "warning":  0.72,
        "normal":   0.08,
    }.get(severity.lower(), 0.08)


def _normalize_severity(severity: str, is_anomaly: bool) -> str:
    """severity를 기존 API 형식으로 정규화 (Normal/Low/Medium/High/Critical)"""
    if not is_anomaly:
        return "Normal"
    return {
        "warning":  "Medium",
        "critical": "High",
    }.get(severity.lower(), "Low")


def _extract_primary_zscore(spc_result: dict) -> tuple[float, str]:
    """SPC 결과에서 1순위 센서 z-score와 센서명 추출"""
    primary = spc_result.get("primary_result")
    if primary and primary.get("z_score") is not None:
        return float(primary["z_score"]), primary.get("sensor", "unknown")
    secondary = spc_result.get("secondary_result")
    if secondary and secondary.get("z_score") is not None:
        return float(secondary["z_score"]), secondary.get("sensor", "unknown")
    return 0.0, "none"


def _infer_fault_type(mode: str, dft_result, spc_result: dict) -> str:
    """고장 유형 추정 (DFT 결과 우선, 없으면 모드·센서 기반)"""
    if dft_result is not None and dft_result.is_anomaly:
        return dft_result.fault_type

    primary = spc_result.get("primary_result")
    if not primary or not primary.get("is_anomaly"):
        # 온도 트렌드 이상인지 확인
        temp_res = spc_result.get("temp_result", {})
        if temp_res and temp_res.get("is_anomaly"):
            return "temperature_trend_anomaly"
        return "normal"

    sensor = primary.get("sensor", "")
    if "Accelerometer" in sensor:
        return "rotor_imbalance_suspected"
    if "Flow" in sensor:
        return {
            "high_flow": "flow_drop_high",
            "mid_flow":  "valve_anomaly_mid",
            "low_flow":  "valve_anomaly_low",
        }.get(mode, "flow_anomaly")
    return "unknown"


def _dft_to_dict(dft_result) -> dict | None:
    """DFTAnomalyResult → 직렬화 가능한 dict"""
    if dft_result is None:
        return None
    return {
        "is_anomaly": dft_result.is_anomaly,
        "A_max":      round(dft_result.A_max,   6),
        "f_max":      round(dft_result.f_max,   4),
        "z_score":    dft_result.z_score,
        "threshold":  dft_result.threshold,
        "severity":   dft_result.severity,
        "fault_type": dft_result.fault_type,
        "detail":     dft_result.detail,
    }


# ─────────────────────────────────────────────────────────────────
# RAAD-LLM LLM-in-the-loop 헬퍼 (논문 Figure 3·4·Eq.8)
# ─────────────────────────────────────────────────────────────────

_SENSOR_KR_ML = {
    "Accelerometer1RMS":   "가속도계1(진동)",
    "Volume_Flow_RateRMS": "유량",
    "Temperature":         "온도",
}

_MODE_KR_ML = {
    "high_flow": "고유량(>100 L/min)",
    "mid_flow":  "중유량(50~100 L/min)",
    "low_flow":  "저유량(<50 L/min)",
}

_THRESHOLDS_ML = {
    "high_flow": {"Accelerometer1RMS": 2.56, "Volume_Flow_RateRMS": 1.71},
    "mid_flow":  {"Volume_Flow_RateRMS": 0.75},
    "low_flow":  {"Volume_Flow_RateRMS": 0.45},
}


def _build_raad_prompt_sync(
    mode: str,
    sensor_zscores: dict,
    temp_trend: dict | None,
    rag_context: dict | None = None,
) -> str:
    """
    논문 Figure 3 형식 프롬프트 (INSTRUCTIONS / DATA / RAG 3-섹션)

    rag_context: retrieve_for_llm() 반환값 (있으면 ChromaDB 검색 결과 사용)
                 없으면 하드코딩 임계값 비교로 fallback
    """
    thresholds = _THRESHOLDS_ML.get(mode, {})

    # ── INSTRUCTIONS ──────────────────────────────────────────────
    questions = [
        f"* Is high deviation present for {_SENSOR_KR_ML.get(s, s)}?"
        for s in sensor_zscores
    ]
    if temp_trend:
        questions.append("* Is high deviation present for 온도(Temperature trend)?")

    instructions = (
        "INSTRUCTIONS: You are a helpful assistant that can use these rules to answer queries. "
        f"The following sensor data was collected over the last 60 seconds from a SKAB water pump "
        f"operating in {_MODE_KR_ML.get(mode, mode)} mode and represents current process conditions. "
        "Strictly based on the DATA and RAG information provided below, please answer the following "
        "questions. Do not modify, interpret, or apply logic beyond these instructions.\n"
        + "\n".join(questions)
        + "\nFor each question, avoid explaining. Just print only the output and nothing else."
    )

    # ── DATA (z-score 실제값) ─────────────────────────────────────
    data_parts = [
        f"{_SENSOR_KR_ML.get(s, s)} has a z-score of {z:.3f}."
        for s, z in sensor_zscores.items()
    ]
    if temp_trend:
        data_parts.append(
            f"온도(Temperature) exhibits a consecutive decrease trend "
            f"over {temp_trend['consec_dec']} data points "
            f"(MA slope = {temp_trend['ma_slope']:.3f}°C/s)."
        )
    data_section = "DATA: " + " ".join(data_parts)

    # ── RAG 섹션 ─────────────────────────────────────────────────
    if rag_context:
        # ChromaDB에서 검색된 실제 도메인 지식 사용 (논문 RAG 단계)
        rag_lines = []
        # 탐지 전략 (모드별 임계값·우선순위)
        for doc in rag_context.get("strategy", [])[:1]:
            rag_lines.append(doc["document"])
        # 임계값·정상범위 문서
        for doc in rag_context.get("thresholds", [])[:2]:
            rag_lines.append(doc["document"])
        # 고장 유형 정보
        for doc in rag_context.get("fault_info", [])[:1]:
            rag_lines.append(doc["document"])
        # 온도 이상 정보 (있을 때만)
        for doc in rag_context.get("temp_info", [])[:1]:
            rag_lines.append(doc["document"])

        rag_section = "RAG: " + " | ".join(rag_lines) if rag_lines else "RAG: No domain knowledge retrieved."
    else:
        # Fallback: 하드코딩 임계값 비교 (ChromaDB 미구축 시)
        rag_parts = []
        for s, z in sensor_zscores.items():
            thr = thresholds.get(s)
            if thr is None:
                continue
            comp = "greater than" if z > thr else ("less than" if z < thr else "equal to")
            rag_parts.append(
                f"The z-score for {_SENSOR_KR_ML.get(s, s)} is {comp} "
                f"acceptable process variable conditions (threshold z={thr})."
            )
        if temp_trend:
            c = temp_trend["consec_dec"]
            comp = "greater than" if c >= 5 else "less than"
            rag_parts.append(
                f"The temperature trend is {comp} acceptable process variable conditions "
                f"(Warning: consec_decrease≥5, Critical: consec_decrease≥10)."
            )
        rag_section = "RAG: " + " ".join(rag_parts) if rag_parts else "RAG: No comparison data."

    return "\n\n".join([instructions, data_section, rag_section])


def _parse_llm_output(llm_output: str) -> list[str]:
    """
    논문 Figure 4 형식 파싱
    입력: "* High deviation is present for 가속도계1(진동).\n* High deviation is not present for 유량."
    출력: ["가속도계1(진동)"]   ← 이상 판정된 센서만 반환
    """
    anomalous = []
    for line in llm_output.strip().split("\n"):
        line = line.strip()
        if not line.startswith("*"):
            continue
        # "is not present" 먼저 확인 (부정이 포함된 경우 제외)
        if "is not present for" in line.lower():
            continue
        if "is present for" in line.lower():
            # "* High deviation is present for X." → X 추출
            idx = line.lower().find("is present for")
            sensor_name = line[idx + len("is present for"):].strip().rstrip(".")
            if sensor_name:
                anomalous.append(sensor_name)
    return anomalous


def _binarize_eq8(mode: str, anomalous_sensors: list[str]) -> int:
    """
    논문 Eq.8: f(x) → {0, 1}
    "f(x) = 1 if anomalies in x are correlated, 0 otherwise"

    SKAB 펌프 도메인 상관관계 규칙:
    - 진동 이상 → 단독으로 유의미 (베어링/임펠러 관련) → f=1
    - 유량 이상 → 단독으로 유의미 (밸브/배관 관련)    → f=1
    - 온도 트렌드 이상 → 밸브 폐쇄 이차 지표          → f=1
    - 이상 없음                                         → f=0
    """
    if not anomalous_sensors:
        return 0

    for s in anomalous_sensors:
        s_lower = s.lower()
        if any(k in s_lower for k in ["진동", "가속도", "accel", "vibration"]):
            return 1
        if any(k in s_lower for k in ["유량", "flow"]):
            return 1
        if any(k in s_lower for k in ["온도", "temp"]):
            return 1

    # 기타 이상 신호 (예: 전류, 압력) → 개수 2개 이상이면 상관관계 의심
    return 1 if len(anomalous_sensors) >= 2 else 0
