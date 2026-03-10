import numpy as np
from datetime import datetime
from collections import deque

class SensorSimulator:
    """
    SKAB 데이터셋 기반 센서 데이터 시뮬레이터
    실시간처럼 1초마다 센서값 생성
    """

    def __init__(self, window_size=60):
        self.window_size = window_size
        # 최근 60개 데이터 버퍼 (WebSocket 실시간 전송용)
        self.buffer = deque(maxlen=window_size)
        self._initialize_buffer()

    def _initialize_buffer(self):
        """버퍼 초기값 채우기"""
        for _ in range(self.window_size):
            self.buffer.append(self._generate_normal())

    def _generate_normal(self) -> dict:
        """정상 범위 센서 데이터 생성"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "accelerometer1": round(np.random.uniform(0.19, 0.23), 4),
            "accelerometer2": round(np.random.uniform(0.25, 0.28), 4),
            "current":        round(np.random.uniform(9.0, 11.0), 3),
            "pressure":       round(np.random.uniform(1.8, 2.2), 3),
            "temperature":    round(np.random.uniform(22.0, 28.0), 2),
            "thermocouple":   round(np.random.uniform(17.0, 23.0), 2),
            "voltage":        round(np.random.uniform(225.0, 235.0), 2),
            "flow_rate":      round(np.random.uniform(9.5, 11.5), 3),
        }

    def _generate_anomaly(self, anomaly_type: str) -> dict:
        """이상 상태 센서 데이터 생성"""
        data = self._generate_normal()

        if anomaly_type == "high_temp":
            data["temperature"]  = round(np.random.uniform(36.0, 45.0), 2)
            data["thermocouple"] = round(np.random.uniform(32.0, 40.0), 2)

        elif anomaly_type == "high_vibration":
            data["accelerometer1"] = round(np.random.uniform(0.8, 1.5), 4)
            data["accelerometer2"] = round(np.random.uniform(0.8, 1.5), 4)

        elif anomaly_type == "low_flow":
            data["flow_rate"] = round(np.random.uniform(4.0, 5.5), 3)
            data["pressure"]  = round(np.random.uniform(0.8, 1.3), 3)

        elif anomaly_type == "high_current":
            data["current"] = round(np.random.uniform(15.0, 18.0), 3)
            data["voltage"] = round(np.random.uniform(200.0, 215.0), 2)

        return data

    def get_next(self) -> dict:
        """
        다음 센서값 반환
        80% 정상 / 20% 이상
        """
        if np.random.random() > 0.2:
            data = self._generate_normal()
        else:
            anomaly_type = np.random.choice([
                "high_temp",
                "high_vibration",
                "low_flow",
                "high_current"
            ])
            data = self._generate_anomaly(anomaly_type)

        self.buffer.append(data)
        return data

    def get_window(self) -> list:
        """
        현재 버퍼의 60개 윈도우를 (60, 8) 형태로 반환
        ML 추론에 사용
        """
        feature_keys = [
            "accelerometer1", "accelerometer2",
            "current", "pressure", "temperature",
            "thermocouple", "voltage", "flow_rate"
        ]
        window = [
            [row[k] for k in feature_keys]
            for row in self.buffer
        ]
        return window

    def get_current(self) -> dict:
        """버퍼의 최신 센서값 반환"""
        if self.buffer:
            return dict(self.buffer[-1])
        return self._generate_normal()
