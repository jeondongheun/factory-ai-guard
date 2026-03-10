import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class TimeSeriesEncoder(nn.Module):
    """LSTM 기반 시계열 인코더"""
    def __init__(self, input_dim=8, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        encoded = self.fc(h_n[-1])
        return encoded


class MLService:
    """ML 모델 서비스 - 서버 시작 시 1회 로드 후 추론만 수행"""

    def __init__(self):
        self.device = "cpu"  # M1 맥북 CPU 추론
        self.model = None
        self.scaler = StandardScaler()
        self.window_size = 60
        self.input_dim = 8
        self.model_path = Path(__file__).parent.parent.parent / "ml" / "weights" / "best_model.pth"
        self.is_loaded = False
        self.model_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    def load_model(self):
        """모델 로드 - 서버 시작 시 1회만 실행"""
        try:
            self.model = TimeSeriesEncoder(
                input_dim=self.input_dim,
                hidden_dim=128,
                num_layers=2
            ).to(self.device)

            if self.model_path.exists():
                checkpoint = torch.load(
                    self.model_path,
                    map_location=self.device
                )
                self.model.load_state_dict(checkpoint['encoder'])

                # 저장된 성능 지표 로드
                if 'metrics' in checkpoint:
                    self.model_metrics = checkpoint['metrics']

                print(f"✅ 모델 로드 완료: {self.model_path}")
            else:
                # .pth 없으면 랜덤 가중치로 초기화 (데모용)
                print("⚠️  best_model.pth 없음 → 랜덤 가중치로 초기화 (데모 모드)")

            self.model.eval()
            self.is_loaded = True

        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            self.is_loaded = False

    def predict(self, sensor_window: list) -> dict:
        """
        센서 윈도우 데이터로 이상 탐지 수행
        sensor_window: (60, 8) 형태의 리스트
        """
        if not self.is_loaded or self.model is None:
            return self._fallback_predict(sensor_window)

        try:
            # numpy 변환 및 정규화
            window_np = np.array(sensor_window)  # (60, 8)
            window_scaled = self.scaler.fit_transform(window_np)

            # 텐서 변환
            window_tensor = torch.FloatTensor(window_scaled)\
                .unsqueeze(0)\
                .to(self.device)  # (1, 60, 8)

            # 추론
            with torch.no_grad():
                encoded = self.model(window_tensor)
                probability = torch.sigmoid(
                    encoded.mean(dim=1, keepdim=True)
                ).item()

            anomaly_detected = probability > 0.5
            severity = self._get_severity(probability)

            return {
                "anomaly_detected": anomaly_detected,
                "probability": round(probability, 4),
                "severity": severity
            }

        except Exception as e:
            print(f"❌ 추론 오류: {e}")
            return self._fallback_predict(sensor_window)

    def _get_severity(self, probability: float) -> str:
        """확률 기반 심각도 분류"""
        if probability >= 0.8:
            return "High"
        elif probability >= 0.6:
            return "Medium"
        elif probability >= 0.5:
            return "Low"
        else:
            return "Normal"

    def _fallback_predict(self, sensor_window: list) -> dict:
        """모델 없을 때 규칙 기반 fallback"""
        window_np = np.array(sensor_window)
        latest = window_np[-1]  # 최신 센서값

        # 간단한 규칙 기반 탐지
        # [accel1, accel2, current, pressure, temp, thermocouple, voltage, flow]
        anomaly_score = 0.0
        if latest[4] > 35:   anomaly_score += 0.4  # 온도
        if latest[0] > 1.0:  anomaly_score += 0.3  # 진동
        if latest[7] < 6:    anomaly_score += 0.3  # 유량

        anomaly_score = min(anomaly_score, 0.99)
        anomaly_detected = anomaly_score > 0.5

        return {
            "anomaly_detected": anomaly_detected,
            "probability": round(anomaly_score, 4),
            "severity": self._get_severity(anomaly_score)
        }

    def get_metrics(self) -> dict:
        """저장된 모델 성능 지표 반환"""
        return self.model_metrics
