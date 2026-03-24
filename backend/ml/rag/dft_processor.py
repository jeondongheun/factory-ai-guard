"""
DFT Processor (Discrete Fourier Transform)
===========================================
논문: RAAD-LLM Section 4.1.3, Equations 4-8

역할:
  SPC 필터로 정화된 진동 시계열을 주파수 도메인으로 변환하여
  지배 주파수(f_max)와 진폭(A_max)을 추출하고,
  baseline C 대비 z-score를 계산해 이상을 탐지한다.

수식 (논문 eq 4-8):
  F(k)  = Σ_{t=0}^{N-1} s(t) · e^{-j·2π·k·t/N}   k=0,...,N/2  (eq 4)
  A_k   = (2/N)|F_k|                                k=0,...,N/2  (eq 5)
  f_max = f_k  where k = argmax_k A_k                            (eq 6)
  A_max = A_k  for same k                                        (eq 7)
  ŝ(t)  = A_max · sin(2π·f_max·t) + |s̄|                        (eq 8)

처리 흐름 (논문 Figure 2):
  원시 진동 → [SPC filter] → 정화된 Q_i
         → 윈도우 분할(P개, 길이 L)
         → 각 윈도우에 DFT → A_max, f_max 추출
         → baseline C 대비 z-score 계산
         → RAG → LLM → {0, 1}

  ※ 이 모듈은 SPC 필터 이후 단계 (DFT ~ z-score까지) 담당
     최종 이상 판정은 spc_processor.ModeAwareZScoreDetector가 수행

고유량 적용:
  - A_max(진동 지배 진폭) 에 z-score > 2.56 → 로터 불균형 탐지
  - 1x 회전 주파수 성분 우세 여부로 이상 유형 식별 (ISO 13373-3)
중/저유량:
  - DFT 불필요. 유량 원시값으로 직접 z-score 계산
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────────────

@dataclass
class DFTResult:
    """단일 윈도우 DFT 분석 결과"""
    # 논문 eq 6-7
    f_max:      float             # 지배 주파수 (Hz 또는 normalized)
    A_max:      float             # 지배 진폭 (eq 7)
    k_max:      int               # argmax 인덱스

    # 스펙트럼 전체
    frequencies: np.ndarray      # 주파수 배열 (eq 5 출력)
    amplitudes:  np.ndarray      # 진폭 스펙트럼 A_k (eq 5)

    # 사인파 재구성 (eq 8)
    reconstructed: np.ndarray    # ŝ(t) = A_max·sin(2π·f_max·t) + |s̄|
    signal_mean:   float         # s̄ (원신호 평균)

    # 통계량
    window_length: int           # 윈도우 길이 N
    snr:           float         # Signal-to-Noise Ratio (dB)


@dataclass
class WindowSetResult:
    """P개 윈도우 전체 DFT 결과"""
    window_results: list[DFTResult]      # 윈도우별 결과
    A_max_series:   np.ndarray           # 윈도우별 A_max 시계열 (shape: P)
    f_max_series:   np.ndarray           # 윈도우별 f_max 시계열 (shape: P)

    # A_max 통계 (baseline 비교용)
    A_max_mean: float
    A_max_std:  float
    A_max_last: float                    # 가장 최근 윈도우 A_max

    n_windows:     int
    window_length: int


@dataclass
class DFTBaseline:
    """
    DFT baseline C (정상 진동의 주파수 특성)
    논문: "baseline dataset C_i initialized as the first Q_i window"
    """
    A_max_values:  list = field(default_factory=list)   # 정상 A_max 누적
    f_max_values:  list = field(default_factory=list)   # 정상 f_max 누적

    A_max_mean: float = 0.0
    A_max_std:  float = 1.0
    f_max_mean: float = 0.0
    f_max_std:  float = 1.0

    is_fitted:    bool = False
    update_count: int  = 0

    def fit(self, window_set: WindowSetResult) -> None:
        """첫 정상 윈도우셋으로 baseline 초기화"""
        A = window_set.A_max_series
        f = window_set.f_max_series

        self.A_max_values = A.tolist()
        self.f_max_values = f.tolist()
        self._recompute_stats()
        self.is_fitted = True

    def update(self, new_A_max: float, new_f_max: float) -> None:
        """
        정상 판정된 윈도우로 적응형 업데이트
        (논문 adaptability: "update C with window Q^(p) if no anomaly")
        """
        self.A_max_values.append(new_A_max)
        self.f_max_values.append(new_f_max)

        # 최근 1000개 유지 (메모리 관리)
        if len(self.A_max_values) > 1000:
            self.A_max_values = self.A_max_values[-1000:]
            self.f_max_values = self.f_max_values[-1000:]

        self._recompute_stats()
        self.update_count += 1

    def _recompute_stats(self) -> None:
        A = np.array(self.A_max_values)
        f = np.array(self.f_max_values)
        self.A_max_mean = float(np.mean(A))
        self.A_max_std  = float(np.std(A, ddof=1)) or 1e-9
        self.f_max_mean = float(np.mean(f))
        self.f_max_std  = float(np.std(f, ddof=1)) or 1e-9

    def zscore_A_max(self, A_max: float) -> float:
        """|z| = |A_max - μ_C| / σ_C"""
        return abs(A_max - self.A_max_mean) / self.A_max_std


@dataclass
class DFTAnomalyResult:
    """DFT 기반 이상 탐지 결과"""
    is_anomaly:   bool
    A_max:        float       # 현재 윈도우 지배 진폭
    f_max:        float       # 현재 윈도우 지배 주파수
    z_score:      float       # A_max z-score vs baseline C
    threshold:    float       # 적용된 임계값
    severity:     str         # 'normal' | 'warning' | 'critical'
    fault_type:   str         # 예상 고장 유형
    detail:       str

    # baseline 정보
    baseline_A_mean: float
    baseline_A_std:  float


# ─────────────────────────────────────────────────────────────────
# 핵심 함수: 단일 윈도우 DFT
# ─────────────────────────────────────────────────────────────────

def apply_dft(
    signal: np.ndarray,
    sample_rate: float = 1.0,
) -> DFTResult:
    """
    단일 윈도우 시계열에 DFT 적용 (논문 eq 4-8)

    Args:
        signal:      1D 시계열 (SPC 필터 완료된 진동값)
        sample_rate: 샘플링 주파수 (Hz). 기본 1.0 = normalized

    Returns:
        DFTResult: f_max, A_max, 재구성 신호, 스펙트럼 전체
    """
    s = np.asarray(signal, dtype=float)
    N = len(s)
    if N < 4:
        raise ValueError(f"DFT에는 최소 4개 포인트 필요 (현재: {N})")

    # ── eq 4: F(k) = Σ s(t)·e^{-j·2π·k·t/N} ────────────────────
    F = np.fft.rfft(s)   # numpy rfft: k=0,...,N//2 (실수 입력 최적화)

    # ── eq 5: A_k = (2/N)|F_k| ──────────────────────────────────
    # k=0 (DC 성분)은 (1/N)|F_0| 으로 처리 (offset 성분)
    A = (2.0 / N) * np.abs(F)
    A[0] = np.abs(F[0]) / N   # DC 보정

    # 주파수 배열
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)

    # ── eq 6: k_max = argmax A_k (DC 성분 제외) ─────────────────
    # k=0(DC)는 평균 오프셋이므로 제외
    A_no_dc    = A.copy()
    A_no_dc[0] = 0.0
    k_max = int(np.argmax(A_no_dc))

    # ── eq 7: f_max, A_max ───────────────────────────────────────
    f_max = float(freqs[k_max])
    A_max = float(A[k_max])

    # ── eq 8: ŝ(t) = A_max·sin(2π·f_max·t) + |s̄| ───────────────
    s_mean = float(np.mean(s))
    t      = np.arange(N) / sample_rate
    if f_max > 0:
        reconstructed = A_max * np.sin(2.0 * np.pi * f_max * t) + abs(s_mean)
    else:
        # 주파수 0인 경우 (DC only) → 평균값으로 채움
        reconstructed = np.full(N, abs(s_mean))

    # SNR 계산: 지배 성분 전력 / 나머지 성분 전력
    dominant_power  = A_max ** 2
    total_power     = float(np.sum(A[1:] ** 2)) or 1e-12
    noise_power     = max(total_power - dominant_power, 1e-12)
    snr_db          = 10.0 * np.log10(dominant_power / noise_power)

    return DFTResult(
        f_max=f_max, A_max=A_max, k_max=k_max,
        frequencies=freqs, amplitudes=A,
        reconstructed=reconstructed, signal_mean=s_mean,
        window_length=N, snr=round(snr_db, 2),
    )


# ─────────────────────────────────────────────────────────────────
# 슬라이딩 윈도우 DFT
# ─────────────────────────────────────────────────────────────────

class SlidingWindowDFT:
    """
    논문 Figure 2: Q_i를 P개 윈도우(길이 L)로 분할 후 DFT

    각 윈도우 Q_i^(p) ∈ R^(1×L)에 독립적으로 DFT 적용.
    A_max 시계열을 추출해 baseline C 대비 통계량을 계산한다.
    """

    def __init__(
        self,
        window_length: int   = 60,    # 윈도우 길이 L (타임스텝 수)
        step_size:     int   = 1,     # 슬라이딩 스텝 (1=매 포인트 갱신)
        sample_rate:   float = 1.0,   # 샘플링 주파수 (Hz)
    ):
        self.window_length = window_length
        self.step_size     = step_size
        self.sample_rate   = sample_rate

    def process_series(self, series: np.ndarray) -> WindowSetResult:
        """
        전체 시계열을 윈도우 분할 후 각 윈도우에 DFT 적용

        Args:
            series: SPC 필터 완료된 1D 진동 시계열

        Returns:
            WindowSetResult: 윈도우별 DFT 결과 + A_max/f_max 시계열
        """
        s = np.asarray(series, dtype=float)
        L = self.window_length

        if len(s) < L:
            raise ValueError(
                f"시계열 길이({len(s)})가 윈도우 길이({L})보다 짧습니다."
            )

        # ── 윈도우 분할 (non-overlapping) ───────────────────────
        starts = range(0, len(s) - L + 1, self.step_size)
        results: list[DFTResult] = []

        for start in starts:
            window = s[start: start + L]
            results.append(apply_dft(window, self.sample_rate))

        A_max_series = np.array([r.A_max for r in results])
        f_max_series = np.array([r.f_max for r in results])

        return WindowSetResult(
            window_results=results,
            A_max_series=A_max_series,
            f_max_series=f_max_series,
            A_max_mean=float(np.mean(A_max_series)),
            A_max_std=float(np.std(A_max_series, ddof=1)) or 1e-9,
            A_max_last=float(A_max_series[-1]),
            n_windows=len(results),
            window_length=L,
        )

    def process_latest_window(self, series: np.ndarray) -> DFTResult:
        """
        가장 최근 윈도우(마지막 L개 포인트)에만 DFT 적용
        (실시간 스트리밍 처리용)

        Args:
            series: 최소 window_length 이상의 1D 진동 시계열

        Returns:
            DFTResult: 최신 윈도우 분석 결과
        """
        s = np.asarray(series, dtype=float)
        if len(s) < self.window_length:
            raise ValueError(
                f"시계열 길이({len(s)}) < 윈도우 길이({self.window_length})"
            )
        window = s[-self.window_length:]
        return apply_dft(window, self.sample_rate)


# ─────────────────────────────────────────────────────────────────
# DFT 이상 탐지기 (A_max vs baseline C)
# ─────────────────────────────────────────────────────────────────

class DFTAnomalyDetector:
    """
    DFT A_max 기반 이상 탐지

    진동 지배 진폭 A_max를 baseline C와 비교해 z-score를 계산.
    고유량 모드: z > 2.56 → 로터 불균형 이상 (CSV id=17, 59)

    ISO 13373-3 기반 고장 유형 분류:
      - 1x 회전주파수 우세 → 로터 불균형
      - 2x 우세           → 축 정렬 불량
      - 광대역 노이즈 증가  → 캐비테이션 또는 베어링 손상
    """

    # 고장 유형별 주파수 패턴 (normalized 기준)
    _FAULT_PATTERNS = {
        "rotor_imbalance": {
            "desc": "로터 불균형 (1x 회전 주파수 우세)",
            "freq_range": (0.01, 0.1),   # normalized 1x 영역
        },
        "misalignment": {
            "desc": "축 정렬 불량 (2x 성분 증가)",
            "freq_range": (0.1, 0.2),
        },
        "bearing_fault": {
            "desc": "베어링 결함 (고주파 성분 증가)",
            "freq_range": (0.2, 0.5),
        },
        "cavitation": {
            "desc": "캐비테이션 (광대역 노이즈)",
            "freq_range": (0.0, 0.5),    # 전대역
        },
    }

    def __init__(
        self,
        z_threshold:        float = 2.56,   # 고유량 1순위 임계값 (CSV id=17)
        critical_multiplier: float = 1.5,   # critical = threshold × 1.5
    ):
        self.z_threshold         = z_threshold
        self.critical_multiplier = critical_multiplier
        self.baseline            = DFTBaseline()
        self._sliding_dft        = SlidingWindowDFT()

    def initialize_baseline(self, normal_series: np.ndarray) -> None:
        """
        정상 진동 데이터로 DFT baseline C 초기화

        Args:
            normal_series: 정상 운영 중 진동 시계열 (SPC 필터 완료)
        """
        window_set = self._sliding_dft.process_series(normal_series)
        self.baseline.fit(window_set)

    def update_baseline(self, A_max: float, f_max: float) -> None:
        """
        정상 판정 후 baseline 적응형 업데이트
        (논문: "update C with Q^(p) if no anomaly detected")
        """
        self.baseline.update(A_max, f_max)

    def detect(
        self,
        signal_window: np.ndarray,
        sample_rate: float = 1.0,
    ) -> DFTAnomalyResult:
        """
        단일 윈도우 DFT 이상 탐지

        Args:
            signal_window: SPC 필터 완료된 진동 윈도우 (길이 L)
            sample_rate:   샘플링 주파수 (Hz)

        Returns:
            DFTAnomalyResult
        """
        if not self.baseline.is_fitted:
            raise RuntimeError(
                "DFT baseline 미초기화. initialize_baseline()을 먼저 호출하세요."
            )

        # DFT 적용
        dft_res = apply_dft(signal_window, sample_rate)

        # z-score 계산
        z = self.baseline.zscore_A_max(dft_res.A_max)

        # 심각도 판정
        crit_thresh = self.z_threshold * self.critical_multiplier
        if z >= crit_thresh:
            severity = "critical"
        elif z >= self.z_threshold:
            severity = "warning"
        else:
            severity = "normal"

        # 고장 유형 분류 (ISO 13373-3)
        fault_type = self._classify_fault(dft_res)

        # 적응형 업데이트 (정상일 때만)
        if severity == "normal":
            self.update_baseline(dft_res.A_max, dft_res.f_max)

        detail = (
            f"A_max={dft_res.A_max:.4f}, f_max={dft_res.f_max:.4f}, "
            f"z={z:.3f}, threshold={self.z_threshold}, "
            f"SNR={dft_res.snr}dB, fault={fault_type}"
        )

        return DFTAnomalyResult(
            is_anomaly=(severity != "normal"),
            A_max=dft_res.A_max,
            f_max=dft_res.f_max,
            z_score=round(z, 4),
            threshold=self.z_threshold,
            severity=severity,
            fault_type=fault_type,
            detail=detail,
            baseline_A_mean=self.baseline.A_max_mean,
            baseline_A_std=self.baseline.A_max_std,
        )

    def _classify_fault(self, dft_res: DFTResult) -> str:
        """
        지배 주파수 위치로 고장 유형 추정 (ISO 13373-3)
        정상 범위면 'normal' 반환
        """
        f = dft_res.f_max
        A = dft_res.amplitudes
        freqs = dft_res.frequencies

        # normalized 주파수 (0~0.5)
        f_norm = f / (dft_res.sample_rate if hasattr(dft_res, "sample_rate") else 1.0)
        # rfftfreq의 최대값이 0.5이므로 이미 normalized
        f_norm = min(f_norm, 0.5)

        # 광대역 vs 협대역 구분
        # 상위 10% 주파수 성분 에너지 비율
        n_top = max(1, len(A) // 10)
        top_k = np.argsort(A[1:])[-n_top:] + 1   # DC 제외
        top_energy_ratio = float(np.sum(A[top_k] ** 2) / (np.sum(A[1:] ** 2) + 1e-12))

        if top_energy_ratio < 0.3:
            # 에너지가 넓게 분산 → 광대역 노이즈 → 캐비테이션
            return "cavitation_broadband"

        if f_norm <= 0.05:
            return "rotor_imbalance_1x"
        elif f_norm <= 0.15:
            return "misalignment_2x"
        elif f_norm <= 0.35:
            return "bearing_fault_hf"
        else:
            return "unknown_hf"


# ─────────────────────────────────────────────────────────────────
# 통합 파이프라인 (SPC filter → DFT → z-score)
# ─────────────────────────────────────────────────────────────────

class VibrationPipeline:
    """
    고유량 모드 진동 이상 탐지 전체 파이프라인

    흐름:
      원시 진동 시계열
        → SPCFilter (2-pass 이상치 제거)
        → SlidingWindowDFT (윈도우별 A_max 추출)
        → DFTAnomalyDetector (baseline z-score 비교)
        → DFTAnomalyResult

    사용 예시:
        pipe = VibrationPipeline()
        pipe.fit_baseline(normal_accel_data)  # 한 번만 호출

        # 실시간 탐지 (60포인트 윈도우)
        result = pipe.detect(accel_window_60pts)
        print(result.is_anomaly, result.z_score, result.fault_type)
    """

    def __init__(
        self,
        window_length: int   = 60,
        z_threshold:   float = 2.56,
        sample_rate:   float = 1.0,
        spc_passes:    int   = 2,
    ):
        self.window_length = window_length
        self.z_threshold   = z_threshold
        self.sample_rate   = sample_rate

        # SPC 필터 (지연 임포트로 순환 의존성 방지)
        from spc_processor import SPCFilter
        self._spc_filter   = SPCFilter(n_passes=spc_passes)
        self._dft_detector = DFTAnomalyDetector(z_threshold=z_threshold)
        self._sliding_dft  = SlidingWindowDFT(
            window_length=window_length,
            sample_rate=sample_rate,
        )

    def fit_baseline(self, normal_series: np.ndarray) -> None:
        """
        정상 진동 데이터로 전체 파이프라인 baseline 초기화

        Args:
            normal_series: 정상 운영 중 원시 진동 데이터 (최소 window_length × 3 권장)
        """
        # 1) SPC 필터로 정화
        spc_result   = self._spc_filter.filter(normal_series)
        clean_series = spc_result.filtered

        # 2) DFT baseline 초기화
        self._dft_detector.initialize_baseline(clean_series)

    def detect(
        self,
        raw_window: np.ndarray,
    ) -> DFTAnomalyResult:
        """
        단일 진동 윈도우 이상 탐지 (실시간용)

        Args:
            raw_window: 원시 진동 윈도우 (길이 ≥ window_length)

        Returns:
            DFTAnomalyResult
        """
        # 1) SPC 필터링 (이상치 제거)
        spc_result    = self._spc_filter.filter(raw_window)
        clean_window  = spc_result.filtered[-self.window_length:]

        # 2) DFT + 이상 탐지
        return self._dft_detector.detect(clean_window, self.sample_rate)

    def get_spectrum(self, raw_window: np.ndarray) -> DFTResult:
        """
        스펙트럼 시각화용 DFT 결과 반환 (이상 판정 없이)

        Args:
            raw_window: 원시 진동 윈도우

        Returns:
            DFTResult: 주파수 스펙트럼 전체
        """
        spc_result   = self._spc_filter.filter(raw_window)
        clean_window = spc_result.filtered[-self.window_length:]
        return apply_dft(clean_window, self.sample_rate)

    @property
    def baseline_info(self) -> dict:
        """baseline 상태 정보"""
        bl = self._dft_detector.baseline
        return {
            "is_fitted":     bl.is_fitted,
            "A_max_mean":    round(bl.A_max_mean, 6),
            "A_max_std":     round(bl.A_max_std,  6),
            "f_max_mean":    round(bl.f_max_mean, 4),
            "update_count":  bl.update_count,
            "n_stored":      len(bl.A_max_values),
        }


# ─────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────

def spectrum_to_dict(dft_res: DFTResult, top_k: int = 5) -> dict:
    """
    DFT 결과를 직렬화 가능한 딕셔너리로 변환 (API 응답용)

    Args:
        dft_res: DFTResult
        top_k:   상위 K개 주파수 성분 포함

    Returns:
        dict: f_max, A_max, top_frequencies, SNR 등
    """
    # 상위 K개 성분 (DC 제외)
    A_no_dc = dft_res.amplitudes.copy()
    A_no_dc[0] = 0.0
    top_idx = np.argsort(A_no_dc)[-top_k:][::-1]

    top_freqs = [
        {
            "rank":      int(i + 1),
            "frequency": round(float(dft_res.frequencies[idx]), 4),
            "amplitude": round(float(dft_res.amplitudes[idx]), 6),
        }
        for i, idx in enumerate(top_idx)
    ]

    return {
        "f_max":          round(dft_res.f_max,       4),
        "A_max":          round(dft_res.A_max,       6),
        "signal_mean":    round(dft_res.signal_mean, 4),
        "snr_db":         dft_res.snr,
        "window_length":  dft_res.window_length,
        "top_frequencies": top_freqs,
    }


# ─────────────────────────────────────────────────────────────────
# 빠른 테스트 (직접 실행 시)
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    # spc_processor 임포트를 위해 경로 추가
    sys.path.insert(0, os.path.dirname(__file__))

    print("=" * 60)
    print("  DFT Processor 기본 동작 테스트")
    print("=" * 60)

    np.random.seed(42)
    SR   = 10.0   # 샘플링 주파수 10Hz (SKAB 약 1~2초 간격이므로 ~1Hz, 여기선 데모)
    N    = 120    # 윈도우 길이

    # ── 1. 단일 윈도우 DFT (논문 eq 4-8) ──────────────────────
    print("\n[ 1 ] 단일 윈도우 DFT (eq 4-8)")

    # 정상: 1Hz 주파수 신호 + 노이즈
    t_norm   = np.arange(N) / SR
    normal_sig = 0.24 * np.sin(2 * np.pi * 1.0 * t_norm) + np.random.normal(0, 0.01, N)

    res_norm = apply_dft(normal_sig, SR)
    print(f"  정상 신호  → f_max={res_norm.f_max:.2f}Hz, A_max={res_norm.A_max:.4f}, SNR={res_norm.snr}dB")
    assert abs(res_norm.f_max - 1.0) < 0.5, f"지배 주파수 오류: {res_norm.f_max}"

    # 이상: 진폭 급증 (로터 불균형 시뮬레이션)
    anomaly_sig = 0.35 * np.sin(2 * np.pi * 1.0 * t_norm) + np.random.normal(0, 0.01, N)
    res_anom    = apply_dft(anomaly_sig, SR)
    print(f"  이상 신호  → f_max={res_anom.f_max:.2f}Hz, A_max={res_anom.A_max:.4f}")
    assert res_anom.A_max > res_norm.A_max, "이상 A_max가 정상보다 작음"
    print("  ✓ eq 4-7 적용 정상 (f_max 탐지, A_max 변화 감지)")

    # ── 2. 사인파 재구성 확인 (eq 8) ───────────────────────────
    print("\n[ 2 ] 사인파 재구성 (eq 8: ŝ(t) = A_max·sin(2πf_max·t) + |s̄|)")
    recon_err = float(np.mean(np.abs(res_norm.reconstructed - normal_sig)))
    print(f"  재구성 평균 오차: {recon_err:.4f} (노이즈 수준이면 정상)")
    assert len(res_norm.reconstructed) == N, "재구성 신호 길이 불일치"
    print("  ✓ 사인파 재구성 완료")

    # ── 3. 슬라이딩 윈도우 DFT ─────────────────────────────────
    print("\n[ 3 ] 슬라이딩 윈도우 DFT (P개 윈도우)")
    long_sig = np.concatenate([
        0.24 * np.sin(2 * np.pi * 1.0 * np.arange(300) / SR) + np.random.normal(0, 0.01, 300),
        0.35 * np.sin(2 * np.pi * 1.0 * np.arange(120) / SR) + np.random.normal(0, 0.01, 120),
    ])

    slider  = SlidingWindowDFT(window_length=N, sample_rate=SR)
    ws_res  = slider.process_series(long_sig)
    print(f"  윈도우 수: {ws_res.n_windows}")
    print(f"  A_max 시계열 평균: {ws_res.A_max_mean:.4f} ± {ws_res.A_max_std:.4f}")
    print(f"  마지막 윈도우 A_max: {ws_res.A_max_last:.4f} (이상 구간이면 높아야 함)")
    assert ws_res.A_max_last > ws_res.A_max_mean, "이상 구간 A_max가 낮음"
    print("  ✓ 슬라이딩 윈도우 처리 정상")

    # ── 4. DFTAnomalyDetector (z-score 이상 탐지) ──────────────
    print("\n[ 4 ] DFT z-score 이상 탐지 (threshold=2.56)")
    detector = DFTAnomalyDetector(z_threshold=2.56)

    # baseline 초기화 (정상 데이터)
    normal_300 = 0.24 * np.sin(2 * np.pi * 1.0 * np.arange(300) / SR) + np.random.normal(0, 0.01, 300)
    detector.initialize_baseline(normal_300)
    print(f"  baseline: A_max_mean={detector.baseline.A_max_mean:.4f}, std={detector.baseline.A_max_std:.4f}")

    # 정상 윈도우 탐지
    r_normal = detector.detect(
        0.24 * np.sin(2 * np.pi * 1.0 * t_norm) + np.random.normal(0, 0.01, N), SR
    )
    print(f"  정상: is_anomaly={r_normal.is_anomaly}, z={r_normal.z_score:.3f}, severity={r_normal.severity}")
    assert not r_normal.is_anomaly

    # 이상 윈도우 탐지 (A_max 3배 = z >> 2.56)
    r_anom = detector.detect(
        0.72 * np.sin(2 * np.pi * 1.0 * t_norm) + np.random.normal(0, 0.01, N), SR
    )
    print(f"  이상: is_anomaly={r_anom.is_anomaly}, z={r_anom.z_score:.3f}, "
          f"fault={r_anom.fault_type}, severity={r_anom.severity}")
    assert r_anom.is_anomaly
    print("  ✓ DFT z-score 이상 탐지 정상")

    # ── 5. VibrationPipeline 통합 테스트 ────────────────────────
    print("\n[ 5 ] VibrationPipeline (SPC→DFT→z-score 통합)")
    pipe = VibrationPipeline(window_length=N, z_threshold=2.56, sample_rate=SR)
    pipe.fit_baseline(normal_300)
    print(f"  baseline info: {pipe.baseline_info}")

    # 스파이크 섞인 이상 신호 탐지
    spike_sig = 0.72 * np.sin(2 * np.pi * 1.0 * t_norm) + np.random.normal(0, 0.01, N)
    spike_sig[30] = 5.0   # SPC가 제거해야 할 스파이크

    r_pipe = pipe.detect(spike_sig)
    print(f"  이상(스파이크 포함): is_anomaly={r_pipe.is_anomaly}, "
          f"z={r_pipe.z_score:.3f}, fault={r_pipe.fault_type}")
    assert r_pipe.is_anomaly

    # spectrum_to_dict 테스트
    spec = pipe.get_spectrum(normal_sig)
    spec_dict = spectrum_to_dict(spec, top_k=3)
    print(f"  스펙트럼 직렬화: {spec_dict}")

    print("\n" + "=" * 60)
    print("  모든 테스트 통과 ✓")
    print("=" * 60)
    sys.exit(0)
