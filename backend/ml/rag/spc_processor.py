"""
SPC Processor (Statistical Process Control)
============================================
논문: RAAD-LLM Section 4.1.3, Figure 1

역할:
  A) 이상치 필터링  : 원시 센서 데이터에서 SPC 관리한계 벗어난 포인트 제거
                     → DFT 입력 데이터 정화 (노이즈·스파이크 제거)
  B) 온도 이상 탐지 : z-score 대신 MAMR 트렌드 기반 탐지
                     → "일정 구간 동안 지속적으로 하락하면 이상"
  C) 적응형 baseline: 이상 없음 판정 시 해당 윈도우를 baseline에 추가
                     → 운영 환경 변화에 자동 적응 (모드별 독립 관리)

MAMR 관리한계 수식 (논문 eq 1-3):
  X Chart:   UCL = X̄ + 2.66·R̄   (eq 1)
             LCL = X̄ - 2.66·R̄   (eq 2)
  mR Chart:  UCL = 3.27·R̄        (eq 3)

  * 2.66 = d2 factor / (d3 factor × √n) → 개별값 관리도 계수
  * 3.27 = D4 factor for n=2 이동범위

SPC 처리 흐름 (논문 Section 4.1.3):
  1) 원시 시계열 Q_i → 1차 SPC 필터 → 이상치 제거
  2) 필터된 데이터로 SPC 재계산 → 2차 SPC 필터
  3) 정화된 Q_i → DFT 처리 (dft_processor.py로 전달)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# 논문 eq 1-3 상수
# ─────────────────────────────────────────────────────────────────
_X_MULT  = 2.66   # X 관리도 UCL/LCL 배수
_MR_MULT = 3.27   # mR 관리도 UCL 배수

# 모드별 탐지 전략 (CSV id=55~57 반영)
MODE_STRATEGY = {
    "high_flow": {
        "use_accel": True,  "accel_z_thresh": 2.56,
        "use_flow":  True,  "flow_z_thresh":  1.71,
        "use_temp":  False,
        "description": "고유량: 진동(1순위 z>2.56) + 유량(2순위 z>1.71)",
    },
    "mid_flow": {
        "use_accel": False, "accel_z_thresh": None,
        "use_flow":  True,  "flow_z_thresh":  0.75,
        "use_temp":  False,
        "description": "중유량: 유량 단독(z>0.75)",
    },
    "low_flow": {
        "use_accel": False, "accel_z_thresh": None,
        "use_flow":  True,  "flow_z_thresh":  0.45,
        "use_temp":  False,
        "description": "저유량: 유량 단독(z>0.45)",
    },
}

# 온도 트렌드 탐지 파라미터 (CSV id=61~62 반영)
TEMP_TREND = {
    "ma_window":              10,      # 이동평균 윈도우 (포인트)
    "consecutive_warn":        5,      # WARNING: N포인트 연속 하락
    "consecutive_critical":   10,      # CRITICAL: N포인트 연속 하락
    "slope_warn":          -0.05,      # WARNING: 이동평균 기울기 °C/s
    "slope_critical":      -0.10,      # CRITICAL: 이동평균 기울기 °C/s
}


# ─────────────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────────────

@dataclass
class MAMRStats:
    """MAMR 차트 통계량 (X 관리도 + mR 관리도)"""
    x_bar:    float   # 이동평균의 전체 평균 (X̄)
    r_bar:    float   # 이동범위의 전체 평균 (R̄)
    ucl_x:    float   # X Chart UCL = X̄ + 2.66·R̄
    lcl_x:    float   # X Chart LCL = X̄ - 2.66·R̄
    ucl_mr:   float   # mR Chart UCL = 3.27·R̄
    n_points: int     # baseline 포인트 수

    def __repr__(self) -> str:
        return (
            f"MAMRStats("
            f"X̄={self.x_bar:.4f}, R̄={self.r_bar:.4f}, "
            f"UCL={self.ucl_x:.4f}, LCL={self.lcl_x:.4f}, "
            f"mR_UCL={self.ucl_mr:.4f}, n={self.n_points})"
        )


@dataclass
class SPCFilterResult:
    """SPC 이상치 필터링 결과"""
    filtered:       np.ndarray   # 이상치 제거된 시계열
    outlier_mask:   np.ndarray   # True = 이상치 (bool array)
    outlier_ratio:  float        # 이상치 비율 (0~1)
    n_outliers:     int          # 이상치 수
    stats:          MAMRStats    # MAMR 통계량
    pass_count:     int          # SPC 통과 횟수 (논문: 2-pass)


@dataclass
class ZScoreResult:
    """모드별 z-score 탐지 결과"""
    is_anomaly:    bool
    sensor:        str            # 'Accelerometer1RMS' | 'Volume_Flow_RateRMS'
    z_score:       float
    threshold:     float
    mode:          str
    value:         float          # 현재값
    baseline_mean: float
    baseline_std:  float
    severity:      str            # 'normal' | 'warning' | 'critical'
    priority:      int            # 1순위 or 2순위


@dataclass
class TemperatureAnomalyResult:
    """온도 트렌드 탐지 결과"""
    is_anomaly:           bool
    trend_direction:      str     # 'decreasing' | 'increasing' | 'stable'
    consecutive_decrease: int     # 연속 하락 포인트 수
    consecutive_increase: int     # 연속 상승 포인트 수
    ma_slope:             float   # 이동평균 기울기 (°C/timestep)
    severity:             str     # 'normal' | 'warning' | 'critical'
    detail:               str


@dataclass
class ModeBaseline:
    """운영 모드별 baseline (적응형 업데이트 지원)"""
    mode:        str
    sensor:      str
    values:      list = field(default_factory=list)   # 정상 데이터 누적
    stats:       Optional[MAMRStats] = None
    mean:        float = 0.0
    std:         float = 1.0
    is_fitted:   bool = False
    update_count: int = 0

    def add_normal_window(self, window: np.ndarray) -> None:
        """정상 윈도우 데이터 추가 (적응형 업데이트 - 논문 eq 6번 스텝)"""
        self.values.extend(window.tolist())
        # 최근 5000 포인트만 유지 (메모리 관리)
        if len(self.values) > 5000:
            self.values = self.values[-5000:]
        self.update_count += 1


# ─────────────────────────────────────────────────────────────────
# 핵심 함수: MAMR 통계량 계산
# ─────────────────────────────────────────────────────────────────

def compute_mamr_stats(series: np.ndarray) -> MAMRStats:
    """
    MAMR 관리도 통계량 계산 (논문 eq 1-3)

    Args:
        series: 1D 시계열 (정상 데이터 기준)

    Returns:
        MAMRStats: X̄, R̄, UCL, LCL, mR UCL
    """
    series = np.asarray(series, dtype=float)
    if len(series) < 2:
        raise ValueError(f"MAMR 계산에는 최소 2개 포인트 필요 (현재: {len(series)})")

    # 이동범위 계산: mR_i = |x_i - x_{i-1}|
    moving_ranges = np.abs(np.diff(series))

    x_bar  = float(np.mean(series))          # X̄: 전체 평균
    r_bar  = float(np.mean(moving_ranges))   # R̄: 이동범위 평균

    # 논문 eq 1-3
    ucl_x  = x_bar + _X_MULT  * r_bar        # UCL = X̄ + 2.66·R̄
    lcl_x  = x_bar - _X_MULT  * r_bar        # LCL = X̄ - 2.66·R̄
    ucl_mr = _MR_MULT * r_bar                 # mR UCL = 3.27·R̄

    return MAMRStats(
        x_bar=x_bar, r_bar=r_bar,
        ucl_x=ucl_x, lcl_x=lcl_x,
        ucl_mr=ucl_mr, n_points=len(series),
    )


# ─────────────────────────────────────────────────────────────────
# SPC 필터 (역할 A - DFT 전처리)
# ─────────────────────────────────────────────────────────────────

class SPCFilter:
    """
    MAMR 기반 이상치 필터 - DFT 입력 데이터 정화
    논문: "SPC is applied again after the first set of outliers are removed" (2-pass)
    """

    def __init__(self, n_passes: int = 2):
        """
        Args:
            n_passes: SPC 필터링 횟수. 논문 기준 2회 (기본값)
        """
        self.n_passes = n_passes

    def filter(self, series: np.ndarray) -> SPCFilterResult:
        """
        n_passes 회 반복 SPC 이상치 제거

        Args:
            series: 원시 센서 시계열 1D array

        Returns:
            SPCFilterResult: 정화된 시계열 + 이상치 마스크 + 통계량
        """
        series = np.asarray(series, dtype=float).copy()
        cumulative_mask = np.zeros(len(series), dtype=bool)  # True = 이상치

        stats = None
        for pass_num in range(1, self.n_passes + 1):
            valid_idx = ~cumulative_mask
            valid_vals = series[valid_idx]

            if len(valid_vals) < 4:
                # 유효 데이터 너무 적으면 중단
                break

            stats = compute_mamr_stats(valid_vals)

            # 관리한계 벗어난 포인트 이상치로 마킹
            out_of_ctrl = (
                (series < stats.lcl_x) |
                (series > stats.ucl_x)
            )
            # mR 기반 이상치 (급격한 변화)
            mr = np.abs(np.diff(series))
            mr_outlier = np.concatenate([[False], mr > stats.ucl_mr])

            pass_mask = out_of_ctrl | mr_outlier
            # 기존에 이미 이상치로 표시된 포인트는 유지
            cumulative_mask = cumulative_mask | pass_mask

        # 이상치 위치를 인접한 정상값으로 보간 (DFT 연속성 유지)
        filtered = _interpolate_outliers(series, cumulative_mask)

        n_out = int(cumulative_mask.sum())
        ratio = n_out / len(series) if len(series) > 0 else 0.0

        return SPCFilterResult(
            filtered=filtered,
            outlier_mask=cumulative_mask,
            outlier_ratio=ratio,
            n_outliers=n_out,
            stats=stats or compute_mamr_stats(series),
            pass_count=self.n_passes,
        )


def _interpolate_outliers(series: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    이상치 위치를 선형 보간으로 채움 (DFT 연속성 유지용)

    Args:
        series: 원시 시계열
        mask:   이상치 위치 (True = 이상치)

    Returns:
        보간된 시계열
    """
    result = series.copy()
    if not mask.any():
        return result

    valid_idx = np.where(~mask)[0]
    if len(valid_idx) < 2:
        # 유효 포인트가 너무 적으면 전체 평균으로 채움
        result[mask] = float(np.nanmean(series[~mask])) if (~mask).any() else 0.0
        return result

    # 선형 보간: np.interp는 범위 밖은 경계값으로 처리
    all_idx = np.arange(len(series))
    result[mask] = np.interp(all_idx[mask], valid_idx, series[valid_idx])
    return result


# ─────────────────────────────────────────────────────────────────
# 온도 트렌드 탐지 (역할 B)
# ─────────────────────────────────────────────────────────────────

class TemperatureTrendDetector:
    """
    온도 트렌드 기반 이상 탐지 (z-score 대신)
    CSV id=61~62, Section 4.1.3 참고

    탐지 원리:
      - 밸브 닫힘 이상: 온도가 천천히 단조 하락 (5~10분간 지속)
      - 베어링 이상:    온도가 지속적으로 상승
      - z-score AUC 0.506 → 탐지 불능 → 트렌드 방식으로 전환

    탐지 방법 (두 가지 병행):
      1) 연속 하락 카운터: 연속 N포인트 하락 시 알람
      2) 이동평균 기울기:  MA slope < threshold 지속 시 알람
    """

    def __init__(
        self,
        ma_window:            int   = TEMP_TREND["ma_window"],
        consecutive_warn:     int   = TEMP_TREND["consecutive_warn"],
        consecutive_critical: int   = TEMP_TREND["consecutive_critical"],
        slope_warn:           float = TEMP_TREND["slope_warn"],
        slope_critical:       float = TEMP_TREND["slope_critical"],
    ):
        self.ma_window            = ma_window
        self.consecutive_warn     = consecutive_warn
        self.consecutive_critical = consecutive_critical
        self.slope_warn           = slope_warn
        self.slope_critical       = slope_critical

        # 내부 상태 (스트리밍 처리용)
        self._buffer: list[float] = []
        self._consec_dec: int     = 0   # 연속 하락 카운터
        self._consec_inc: int     = 0   # 연속 상승 카운터

    def reset(self) -> None:
        """상태 초기화 (모드 전환 시 호출)"""
        self._buffer     = []
        self._consec_dec = 0
        self._consec_inc = 0

    def update(self, value: float) -> TemperatureAnomalyResult:
        """
        새 온도 포인트 처리 (스트리밍 방식)

        Args:
            value: 현재 온도값 (°C)

        Returns:
            TemperatureAnomalyResult
        """
        self._buffer.append(value)
        if len(self._buffer) > max(self.ma_window * 3, 50):
            self._buffer = self._buffer[-max(self.ma_window * 3, 50):]

        # ── 연속 하락/상승 카운터 업데이트 ──────────────────────
        if len(self._buffer) >= 2:
            if self._buffer[-1] < self._buffer[-2]:
                self._consec_dec += 1
                self._consec_inc  = 0
            elif self._buffer[-1] > self._buffer[-2]:
                self._consec_inc += 1
                self._consec_dec  = 0
            else:
                # 동일값: 카운터 유지 (소폭 변동 허용)
                pass

        # ── 이동평균 기울기 계산 ─────────────────────────────────
        ma_slope = 0.0
        if len(self._buffer) >= self.ma_window:
            window = np.array(self._buffer[-self.ma_window:])
            ma = float(np.mean(window))
            # 이전 윈도우 평균과 비교해 기울기 근사
            if len(self._buffer) >= self.ma_window + 1:
                prev_window = np.array(
                    self._buffer[-(self.ma_window + 1):-1]
                )
                prev_ma = float(np.mean(prev_window))
                ma_slope = ma - prev_ma   # °C / timestep

        # ── 트렌드 방향 결정 ─────────────────────────────────────
        if self._consec_dec >= 2:
            trend = "decreasing"
        elif self._consec_inc >= 2:
            trend = "increasing"
        else:
            trend = "stable"

        # ── 심각도 판정 ──────────────────────────────────────────
        # 하락 트렌드 (밸브 이상)
        dec_critical = (
            self._consec_dec >= self.consecutive_critical or
            ma_slope <= self.slope_critical
        )
        dec_warning = (
            self._consec_dec >= self.consecutive_warn or
            ma_slope <= self.slope_warn
        )

        # 상승 트렌드 (베어링 이상 등)
        inc_critical = (
            self._consec_inc >= self.consecutive_critical or
            ma_slope >= -self.slope_critical   # 상승은 양수 slope
        )
        inc_warning = (
            self._consec_inc >= self.consecutive_warn or
            ma_slope >= -self.slope_warn
        )

        if dec_critical or inc_critical:
            severity   = "critical"
            is_anomaly = True
        elif dec_warning or inc_warning:
            severity   = "warning"
            is_anomaly = True
        else:
            severity   = "normal"
            is_anomaly = False

        # ── 상세 설명 생성 ───────────────────────────────────────
        detail = (
            f"트렌드={trend}, "
            f"연속하락={self._consec_dec}포인트, "
            f"연속상승={self._consec_inc}포인트, "
            f"MA기울기={ma_slope:+.4f}°C/step, "
            f"심각도={severity}"
        )

        return TemperatureAnomalyResult(
            is_anomaly=is_anomaly,
            trend_direction=trend,
            consecutive_decrease=self._consec_dec,
            consecutive_increase=self._consec_inc,
            ma_slope=ma_slope,
            severity=severity,
            detail=detail,
        )

    def detect_batch(self, series: np.ndarray) -> TemperatureAnomalyResult:
        """
        배치 시계열 처리 (전체 윈도우 한번에 분석)

        Args:
            series: 온도 시계열 1D array

        Returns:
            마지막 포인트의 TemperatureAnomalyResult
        """
        self.reset()
        result = None
        for val in np.asarray(series, dtype=float):
            result = self.update(float(val))
        return result


# ─────────────────────────────────────────────────────────────────
# z-score 계산 (모드별 baseline 기반)
# ─────────────────────────────────────────────────────────────────

class ModeAwareZScoreDetector:
    """
    모드별 baseline 분리 z-score 이상 탐지
    CSV id=55~57 전략 구현:
      고유량: Accelerometer1RMS z > 2.56 (1순위) + Volume Flow Rate z > 1.71 (2순위)
      중유량: Volume Flow Rate z > 0.75 (단독)
      저유량: Volume Flow Rate z > 0.45 (단독)
    """

    def __init__(self):
        # {mode: {sensor: ModeBaseline}}
        self._baselines: dict[str, dict[str, ModeBaseline]] = {
            mode: {} for mode in MODE_STRATEGY
        }

    def fit(
        self,
        mode: str,
        sensor: str,
        normal_data: np.ndarray,
    ) -> None:
        """
        모드·센서별 baseline 초기화

        Args:
            mode:        'high_flow' | 'mid_flow' | 'low_flow'
            sensor:      'Accelerometer1RMS' | 'Volume_Flow_RateRMS'
            normal_data: 정상 데이터 1D array
        """
        data = np.asarray(normal_data, dtype=float)
        data = data[np.isfinite(data)]
        if len(data) < 2:
            raise ValueError(f"baseline 학습에 최소 2개 포인트 필요 (mode={mode}, sensor={sensor})")

        baseline           = ModeBaseline(mode=mode, sensor=sensor)
        baseline.values    = data.tolist()
        baseline.mean      = float(np.mean(data))
        baseline.std       = float(np.std(data, ddof=1)) or 1e-9
        baseline.stats     = compute_mamr_stats(data)
        baseline.is_fitted = True

        self._baselines[mode][sensor] = baseline

    def update_baseline(self, mode: str, sensor: str, normal_window: np.ndarray) -> None:
        """
        정상 윈도우로 baseline 적응형 업데이트 (논문 adaptability mechanism)

        Args:
            mode:         운영 모드
            sensor:       센서명
            normal_window: 정상 판정된 윈도우 데이터
        """
        if mode not in self._baselines or sensor not in self._baselines[mode]:
            # 아직 fit 안 된 경우 직접 초기화
            self.fit(mode, sensor, normal_window)
            return

        bl = self._baselines[mode][sensor]
        bl.add_normal_window(np.asarray(normal_window, dtype=float))

        # mean / std 재계산
        data = np.array(bl.values)
        bl.mean = float(np.mean(data))
        bl.std  = float(np.std(data, ddof=1)) or 1e-9
        if len(data) >= 2:
            bl.stats = compute_mamr_stats(data)

    def compute_zscore(self, mode: str, sensor: str, value: float) -> float:
        """
        단일값 z-score 계산 (모드별 baseline 기준)

        Args:
            mode:   운영 모드
            sensor: 센서명
            value:  현재 센서값

        Returns:
            |z-score| (절대값)
        """
        if (mode not in self._baselines or
                sensor not in self._baselines[mode] or
                not self._baselines[mode][sensor].is_fitted):
            raise RuntimeError(
                f"baseline 미학습: mode={mode}, sensor={sensor}. "
                f"먼저 fit()을 호출하세요."
            )

        bl = self._baselines[mode][sensor]
        return abs((value - bl.mean) / bl.std)

    def detect(
        self,
        mode: str,
        sensor: str,
        value: float,
        priority: int = 1,
    ) -> ZScoreResult:
        """
        단일값 이상 탐지 (z-score > 모드별 임계값)

        Args:
            mode:     운영 모드
            sensor:   센서명 ('Accelerometer1RMS' | 'Volume_Flow_RateRMS')
            value:    현재 센서값
            priority: 1 = 1순위, 2 = 2순위

        Returns:
            ZScoreResult
        """
        strategy = MODE_STRATEGY.get(mode, {})

        # 임계값 결정
        if "Accelerometer" in sensor:
            threshold = strategy.get("accel_z_thresh")
            if threshold is None:
                return ZScoreResult(
                    is_anomaly=False, sensor=sensor,
                    z_score=0.0, threshold=0.0, mode=mode,
                    value=value, baseline_mean=0.0, baseline_std=1.0,
                    severity="normal", priority=priority,
                )
        else:  # Volume Flow Rate
            threshold = strategy.get("flow_z_thresh", 1.0)

        z = self.compute_zscore(mode, sensor, value)
        bl = self._baselines[mode][sensor]

        # 심각도 판정 (임계값 1.5배 = critical)
        if z >= threshold * 1.5:
            severity = "critical"
        elif z >= threshold:
            severity = "warning"
        else:
            severity = "normal"

        return ZScoreResult(
            is_anomaly=(z >= threshold),
            sensor=sensor,
            z_score=round(z, 4),
            threshold=threshold,
            mode=mode,
            value=value,
            baseline_mean=bl.mean,
            baseline_std=bl.std,
            severity=severity,
            priority=priority,
        )

    def detect_window(
        self,
        mode: str,
        sensor: str,
        window: np.ndarray,
        priority: int = 1,
    ) -> ZScoreResult:
        """
        윈도우 평균값으로 이상 탐지 (배치 처리용)

        Args:
            mode:     운영 모드
            sensor:   센서명
            window:   센서 윈도우 1D array
            priority: 순위

        Returns:
            ZScoreResult (윈도우 평균 기준)
        """
        mean_val = float(np.mean(np.asarray(window, dtype=float)))
        return self.detect(mode, sensor, mean_val, priority)

    def get_baseline_info(self, mode: str, sensor: str) -> dict:
        """baseline 상태 정보 반환"""
        if mode not in self._baselines or sensor not in self._baselines[mode]:
            return {"fitted": False}
        bl = self._baselines[mode][sensor]
        return {
            "fitted":       bl.is_fitted,
            "mean":         round(bl.mean, 4),
            "std":          round(bl.std, 4),
            "n_points":     len(bl.values),
            "update_count": bl.update_count,
            "stats":        str(bl.stats) if bl.stats else None,
        }


# ─────────────────────────────────────────────────────────────────
# 통합 SPC Processor (세 역할 통합)
# ─────────────────────────────────────────────────────────────────

class SPCProcessor:
    """
    RAAD-LLM SPC 통합 처리기

    역할 A: SPCFilter         → 원시 데이터 이상치 제거 (DFT 전처리)
    역할 B: TemperatureTrendDetector → 온도 트렌드 탐지
    역할 C: ModeAwareZScoreDetector  → 모드별 z-score 탐지

    사용 예시:
        proc = SPCProcessor()

        # baseline 학습
        proc.fit_baseline('high_flow', 'Accelerometer1RMS', normal_accel_data)
        proc.fit_baseline('high_flow', 'Volume_Flow_RateRMS', normal_flow_data)

        # 실시간 탐지
        result = proc.process(
            mode='high_flow',
            accel_window=accel_data[-60:],
            flow_value=current_flow,
            temp_value=current_temp,
        )
    """

    def __init__(self, spc_passes: int = 2):
        self.spc_filter    = SPCFilter(n_passes=spc_passes)
        self.temp_detector = TemperatureTrendDetector()
        self.zscore_det    = ModeAwareZScoreDetector()
        self._current_mode = None

    def fit_baseline(
        self,
        mode: str,
        sensor: str,
        normal_data: np.ndarray,
    ) -> None:
        """모드·센서별 baseline 학습 (위임)"""
        self.zscore_det.fit(mode, sensor, normal_data)

    def filter_for_dft(self, raw_series: np.ndarray) -> SPCFilterResult:
        """
        역할 A: DFT 전처리용 이상치 필터링

        Args:
            raw_series: 원시 진동 시계열

        Returns:
            SPCFilterResult (정화된 시계열 포함)
        """
        return self.spc_filter.filter(raw_series)

    def detect_temperature(self, temp_value: float) -> TemperatureAnomalyResult:
        """
        역할 B: 온도 트렌드 이상 탐지 (스트리밍)

        Args:
            temp_value: 현재 온도값 (°C)

        Returns:
            TemperatureAnomalyResult
        """
        return self.temp_detector.update(temp_value)

    def detect_temperature_batch(self, temp_series: np.ndarray) -> TemperatureAnomalyResult:
        """역할 B: 온도 배치 탐지"""
        return self.temp_detector.detect_batch(temp_series)

    def detect_zscore(
        self,
        mode: str,
        sensor: str,
        value: float,
        priority: int = 1,
    ) -> ZScoreResult:
        """
        역할 C: 모드별 z-score 탐지 (단일값)

        Args:
            mode:     운영 모드
            sensor:   센서명
            value:    현재값
            priority: 1 or 2

        Returns:
            ZScoreResult
        """
        return self.zscore_det.detect(mode, sensor, value, priority)

    def update_baseline(
        self,
        mode: str,
        sensor: str,
        normal_window: np.ndarray,
    ) -> None:
        """
        역할 C: 정상 판정 윈도우로 baseline 적응형 업데이트

        Args:
            mode:          운영 모드
            sensor:        센서명
            normal_window: 정상 판정된 데이터 윈도우
        """
        self.zscore_det.update_baseline(mode, sensor, normal_window)

    def on_mode_change(self, new_mode: str) -> None:
        """
        모드 전환 시 호출 (온도 트렌드 카운터 리셋)

        Args:
            new_mode: 새 운영 모드
        """
        if self._current_mode != new_mode:
            self.temp_detector.reset()
            self._current_mode = new_mode

    def process(
        self,
        mode: str,
        flow_value: float,
        temp_value: float,
        accel_window: Optional[np.ndarray] = None,
        update_on_normal: bool = True,
    ) -> dict:
        """
        통합 처리: 모드별 전략에 따라 전체 탐지 수행

        Args:
            mode:             현재 운영 모드 ('high_flow' | 'mid_flow' | 'low_flow')
            flow_value:       현재 유량값 (L/min)
            temp_value:       현재 온도값 (°C)
            accel_window:     진동 윈도우 (고유량에서만 필요, shape=(L,))
            update_on_normal: 정상 판정 시 baseline 자동 업데이트 여부

        Returns:
            dict:
                mode          : 현재 모드
                is_anomaly    : 최종 이상 여부
                severity      : 'normal' | 'warning' | 'critical'
                primary_result: 1순위 탐지 결과
                secondary_result: 2순위 탐지 결과 (고유량만)
                temp_result   : 온도 트렌드 결과 (보조)
                details       : 상세 딕셔너리
        """
        # 모드 전환 처리
        self.on_mode_change(mode)
        strategy = MODE_STRATEGY.get(mode, MODE_STRATEGY["low_flow"])

        result = {
            "mode":             mode,
            "is_anomaly":       False,
            "severity":         "normal",
            "primary_result":   None,
            "secondary_result": None,
            "temp_result":      None,
            "details":          {"strategy": strategy["description"]},
        }

        # ── 온도 트렌드 (보조, 모든 모드) ────────────────────────
        temp_result = self.detect_temperature(temp_value)
        result["temp_result"] = {
            "is_anomaly":  temp_result.is_anomaly,
            "trend":       temp_result.trend_direction,
            "consec_dec":  temp_result.consecutive_decrease,
            "ma_slope":    round(temp_result.ma_slope, 4),
            "severity":    temp_result.severity,
            "detail":      temp_result.detail,
        }

        # ── 고유량 모드: 진동(1순위) + 유량(2순위) ───────────────
        if mode == "high_flow":
            primary_anom  = False
            secondary_anom = False

            # 1순위: Accelerometer1RMS
            if accel_window is not None and len(accel_window) > 0:
                try:
                    # SPC 필터로 이상치 제거 후 평균
                    filtered = self.filter_for_dft(accel_window)
                    accel_mean = float(np.mean(filtered.filtered))
                    prim = self.detect_zscore(mode, "Accelerometer1RMS", accel_mean, priority=1)
                    result["primary_result"] = {
                        "sensor":    prim.sensor,
                        "z_score":   prim.z_score,
                        "threshold": prim.threshold,
                        "value":     round(accel_mean, 4),
                        "is_anomaly": prim.is_anomaly,
                        "severity":  prim.severity,
                        "priority":  1,
                    }
                    primary_anom = prim.is_anomaly
                    if prim.is_anomaly and update_on_normal is False:
                        pass
                    elif not prim.is_anomaly and update_on_normal:
                        self.update_baseline(mode, "Accelerometer1RMS", accel_window)
                except RuntimeError as e:
                    result["details"]["accel_error"] = str(e)

            # 2순위: Volume Flow Rate
            try:
                sec = self.detect_zscore(mode, "Volume_Flow_RateRMS", flow_value, priority=2)
                result["secondary_result"] = {
                    "sensor":    sec.sensor,
                    "z_score":   sec.z_score,
                    "threshold": sec.threshold,
                    "value":     flow_value,
                    "is_anomaly": sec.is_anomaly,
                    "severity":  sec.severity,
                    "priority":  2,
                }
                secondary_anom = sec.is_anomaly
                if not sec.is_anomaly and update_on_normal:
                    self.update_baseline(mode, "Volume_Flow_RateRMS", np.array([flow_value]))
            except RuntimeError as e:
                result["details"]["flow_error"] = str(e)

            # 최종 판정: 1순위 OR (1순위 없음 + 2순위)
            final_anom = primary_anom or secondary_anom

        # ── 중/저유량 모드: 유량 단독 ────────────────────────────
        else:
            final_anom = False
            try:
                prim = self.detect_zscore(mode, "Volume_Flow_RateRMS", flow_value, priority=1)
                result["primary_result"] = {
                    "sensor":    prim.sensor,
                    "z_score":   prim.z_score,
                    "threshold": prim.threshold,
                    "value":     flow_value,
                    "is_anomaly": prim.is_anomaly,
                    "severity":  prim.severity,
                    "priority":  1,
                }
                final_anom = prim.is_anomaly
                if not prim.is_anomaly and update_on_normal:
                    self.update_baseline(mode, "Volume_Flow_RateRMS", np.array([flow_value]))
            except RuntimeError as e:
                result["details"]["flow_error"] = str(e)

        # ── 온도 이상 시 severity 격상 (보조 역할) ───────────────
        if temp_result.is_anomaly and temp_result.severity == "critical":
            # 온도 critical이면 최종 결과도 격상
            final_anom = True

        # ── severity 결정 ─────────────────────────────────────────
        severities = []
        if result["primary_result"]:
            severities.append(result["primary_result"]["severity"])
        if result["secondary_result"]:
            severities.append(result["secondary_result"]["severity"])
        if temp_result.is_anomaly:
            severities.append(temp_result.severity)

        if "critical" in severities:
            final_severity = "critical"
        elif "warning" in severities:
            final_severity = "warning"
        else:
            final_severity = "normal"

        result["is_anomaly"] = final_anom
        result["severity"]   = final_severity if final_anom else "normal"

        return result


# ─────────────────────────────────────────────────────────────────
# 유틸리티: 유량으로 모드 판별
# ─────────────────────────────────────────────────────────────────

def get_operating_mode(flow_rate: float) -> str:
    """
    유량값으로 운영 모드 판별 (CSV id=18 기준)

    Args:
        flow_rate: Volume Flow Rate (L/min)

    Returns:
        'high_flow' | 'mid_flow' | 'low_flow'
    """
    if flow_rate > 100:
        return "high_flow"
    elif flow_rate > 50:
        return "mid_flow"
    else:
        return "low_flow"


# ─────────────────────────────────────────────────────────────────
# 빠른 테스트 (직접 실행 시)
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  SPC Processor 기본 동작 테스트")
    print("=" * 60)

    np.random.seed(42)

    # ── 1. MAMR 통계량 계산 테스트 ─────────────────────────────
    print("\n[ 1 ] MAMR 통계량 계산 (eq 1-3)")
    normal_accel = np.random.normal(0.24, 0.02, 300)   # 고유량 정상 진동
    stats = compute_mamr_stats(normal_accel)
    print(f"  {stats}")
    assert stats.ucl_x > stats.x_bar > stats.lcl_x, "UCL > X̄ > LCL 조건 실패"
    print("  ✓ UCL > X̄ > LCL 조건 통과")

    # ── 2. SPC 필터 테스트 ─────────────────────────────────────
    print("\n[ 2 ] SPC 2-pass 이상치 필터")
    data_with_spikes = normal_accel.copy()
    spike_positions = [50, 100, 200]
    for pos in spike_positions:
        data_with_spikes[pos] = 5.0   # 스파이크 주입
    filt_result = SPCFilter(n_passes=2).filter(data_with_spikes)
    print(f"  이상치 수: {filt_result.n_outliers} / {len(data_with_spikes)}")
    print(f"  이상치 비율: {filt_result.outlier_ratio:.2%}")
    print(f"  이상치 위치: {np.where(filt_result.outlier_mask)[0].tolist()[:10]}")
    detected = set(np.where(filt_result.outlier_mask)[0])
    for pos in spike_positions:
        assert pos in detected, f"스파이크 위치 {pos} 미탐지"
    print("  ✓ 주입된 스파이크 모두 탐지")

    # ── 3. 온도 트렌드 탐지 테스트 ─────────────────────────────
    print("\n[ 3 ] 온도 트렌드 탐지")
    detector = TemperatureTrendDetector()

    # 정상 구간 (안정)
    for t in np.random.normal(85, 0.5, 20):
        r = detector.update(float(t))
    print(f"  정상 구간: severity={r.severity}, consec_dec={r.consecutive_decrease}")
    assert not r.is_anomaly, "정상 구간에서 이상 탐지 오작동"

    # 밸브 이상: 지속 하락
    temp_vals = np.linspace(85, 74, 15)   # 85°C → 74°C 선형 하락
    for t in temp_vals:
        r = detector.update(float(t))
    print(f"  하락 구간: severity={r.severity}, consec_dec={r.consecutive_decrease}, "
          f"slope={r.ma_slope:+.4f}")
    assert r.is_anomaly, "온도 하락 이상 미탐지"
    print("  ✓ 온도 지속 하락 이상 탐지 성공")

    # ── 4. 모드별 z-score 탐지 테스트 ─────────────────────────
    print("\n[ 4 ] 모드별 z-score 탐지")
    det = ModeAwareZScoreDetector()

    # baseline 학습 (고유량 정상 데이터)
    det.fit("high_flow", "Accelerometer1RMS", normal_accel)
    det.fit("high_flow", "Volume_Flow_RateRMS", np.random.normal(124, 2, 300))
    det.fit("mid_flow",  "Volume_Flow_RateRMS", np.random.normal(75, 2, 300))
    det.fit("low_flow",  "Volume_Flow_RateRMS", np.random.normal(32, 1, 300))

    # 고유량: 진동 정상
    r1 = det.detect("high_flow", "Accelerometer1RMS", 0.25, priority=1)
    print(f"  고유량 진동 정상값: z={r1.z_score:.3f}, is_anomaly={r1.is_anomaly}")
    assert not r1.is_anomaly

    # 고유량: 진동 이상 (z > 2.56)
    r2 = det.detect("high_flow", "Accelerometer1RMS", 0.24 + 2.8 * 0.02, priority=1)
    print(f"  고유량 진동 이상값: z={r2.z_score:.3f}, is_anomaly={r2.is_anomaly}, severity={r2.severity}")
    assert r2.is_anomaly

    # 저유량: 유량 이상 (z > 0.45)
    r3 = det.detect("low_flow", "Volume_Flow_RateRMS", 32 - 0.5 * 1, priority=1)
    print(f"  저유량 유량 이상값: z={r3.z_score:.3f}, is_anomaly={r3.is_anomaly}, threshold={r3.threshold}")
    assert r3.is_anomaly, "저유량 유량 이상 미탐지"
    print("  ✓ 모드별 z-score 탐지 정상")

    # ── 5. 통합 프로세서 테스트 ─────────────────────────────────
    print("\n[ 5 ] SPCProcessor 통합 테스트")
    proc = SPCProcessor()
    proc.fit_baseline("high_flow", "Accelerometer1RMS", normal_accel)
    proc.fit_baseline("high_flow", "Volume_Flow_RateRMS", np.random.normal(124, 2, 300))
    proc.fit_baseline("low_flow",  "Volume_Flow_RateRMS", np.random.normal(32, 1, 300))

    # 고유량 정상
    res = proc.process(
        mode="high_flow",
        flow_value=124.0,
        temp_value=87.0,
        accel_window=np.random.normal(0.24, 0.02, 60),
    )
    print(f"  고유량 정상: is_anomaly={res['is_anomaly']}, severity={res['severity']}")

    # 저유량 이상 (유량 급감)
    res2 = proc.process(
        mode="low_flow",
        flow_value=31.5,   # z ≈ 0.5 > 0.45
        temp_value=72.0,
    )
    print(f"  저유량 이상: is_anomaly={res2['is_anomaly']}, "
          f"z={res2['primary_result']['z_score'] if res2['primary_result'] else 'N/A'}")

    print("\n" + "=" * 60)
    print("  모든 테스트 통과 ✓")
    print("=" * 60)
    sys.exit(0)
