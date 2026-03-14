# Factory-AI-Guard

> **Real-time smart factory anomaly detection system**  
> LSTM time-series encoder + RAG-augmented LLM diagnosis | FastAPI · React · Docker Compose

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)

---

## 프로젝트 개요

스마트팩토리 환경에서 8개 센서 데이터를 실시간으로 수집·분석하여 설비 이상을 자동으로 탐지하고, RAG 기반 LLM이 원인 분석 및 조치 방안을 제시하는 End-to-End AI 모니터링 시스템.

### 핵심 기술
- LSTM 시계열 인코더: 60개 타임스텝 윈도우 기반 이상 패턴 학습
- RAG + LLM 진단: 유지보수 매뉴얼을 벡터 검색하여 LLM 프롬프트에 주입, Claude API로 원인 분석
- 실시간 WebSocket 스트리밍: 1초 간격 센서 데이터 푸시
- REST API: FastAPI 기반 16개 엔드포인트, Swagger 자동 문서화
- Docker Compose: 멀티컨테이너 서비스 운영 (Backend / Frontend / PostgreSQL / Redis / Nginx)

---

## 사용 데이터셋

| 데이터셋 | 설명 | 센서 수 | 출처 |
|---------|------|--------|------|
| **SKAB** | 스마트팩토리 펌프 시스템 | 8개 (가속도계, 온도, 압력, 유량 등) | [GitHub](https://github.com/waico/SKAB) |

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    React Dashboard                       │
│         실시간 차트 · 알람 · LLM 진단 · 통계             │
└──────────────┬──────────────────────┬───────────────────┘
               │ REST API             │ WebSocket
               ▼                      ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ LSTM 추론   │  │ Claude API  │  │  Sensor          │ │
│  │ (CPU 추론)  │  │ RAG + LLM   │  │  Simulator       │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└──────────┬──────────────────────────────────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
PostgreSQL      Redis
(이력 저장)    (캐시)
```

---

## 🔌 REST API 명세

| Method | Endpoint | 설명 |
|--------|----------|------|
| `GET` | `/api/sensors/current` | 현재 센서값 조회 |
| `GET` | `/api/sensors/history` | 센서 이력 조회 |
| `WS` | `/ws/realtime` | 실시간 WebSocket 스트림 |
| `POST` | `/api/detection/analyze` | 이상 탐지 분석 요청 |
| `GET` | `/api/detection/history` | 탐지 이력 조회 |
| `POST` | `/api/diagnosis/llm` | LLM 상세 진단 요청 |
| `GET` | `/api/stats/summary` | 통계 요약 |
| `GET` | `/api/stats/trend` | 기간별 트렌드 |
| `GET` | `/api/model/metrics` | 모델 성능 지표 |
| `POST` | `/api/upload/csv` | CSV 배치 분석 |
| `PUT` | `/api/settings/thresholds` | 임계값 설정 |

> 전체 API 문서: `http://localhost:8000/docs` (Swagger UI 자동 생성)

---

## 주요 기능

### 1. 실시간 대시보드
- 8채널 센서값 카드 (정상/경고/위험 색상 구분)
- WebSocket 기반 실시간 라인차트 (최근 60초)
- 이상 감지 즉시 알람 배너

### 2. LLM 진단 패널
- 이상 감지 시 Claude API 호출
- RAG로 유지보수 매뉴얼 검색 후 프롬프트 주입
- 원인 분석 + 권장 조치 한국어 출력

### 3. 탐지 이력
- 날짜/심각도 필터링
- 페이지네이션

### 4. 통계 페이지
- 일별 이상 발생 추이 바차트
- 정상/이상 비율 파이차트
- 센서별 평균값 시각화

### 5. 모델 성능 지표
- Accuracy / Precision / Recall / F1 Score
- 레이더 차트 시각화
- 모델 구조 정보 (LSTM layer, hidden dim 등)

### 6. CSV 배치 분석
- SKAB 형식 CSV 업로드
- 전체 구간 이상 확률 차트 출력

### 7. 임계값 설정
- 센서별 경고/위험 기준값 커스텀
- DB 저장 및 실시간 반영

---

## 기술 스택

### Backend
| 기술 | 용도 |
|------|------|
| FastAPI | REST API + WebSocket 서버 |
| SQLAlchemy (async) | ORM (PostgreSQL 비동기 연결) |
| PyTorch | LSTM 모델 추론 |
| Anthropic SDK | Claude API (LLM 진단) |
| Redis | 실시간 데이터 캐시 |

### Frontend
| 기술 | 용도 |
|------|------|
| React 18 | UI 프레임워크 |
| Recharts | 센서 데이터 시각화 |
| React Router | SPA 라우팅 |
| TailwindCSS | 스타일링 |
| Axios | REST API 통신 |

### Infrastructure
| 기술 | 용도 |
|------|------|
| Docker Compose | 멀티컨테이너 오케스트레이션 |
| PostgreSQL 15 | 탐지 이력 / 설정값 저장 |
| Nginx | 리버스 프록시 |

---

## 실행 방법

### 사전 요구사항
- Docker Desktop
- Node.js 18+
- Python 3.9+

### 1. 레포지토리 클론
```bash
git clone https://github.com/{username}/factory-ai-guard.git
cd factory-ai-guard
```

### 2. 환경변수 설정
```bash
cp .env.example .env
```

`.env` 파일에 API 키 입력:
```env
ANTHROPIC_API_KEY=your_claude_api_key_here
POSTGRES_USER=factoryguard
POSTGRES_PASSWORD=factoryguard1234
POSTGRES_DB=factoryguard_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. Docker로 DB 실행
```bash
docker-compose up db redis -d
```

### 4. 백엔드 실행
```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 5. 프론트엔드 실행
```bash
cd frontend
npm install
npm start
```

### 6. 접속
| 서비스 | URL |
|--------|-----|
| 대시보드 | http://localhost:3000 |
| API 문서 | http://localhost:8000/docs |

---

## 모델 학습 (선택)

모델은 Google Colab (GPU) 환경에서 학습 후 `.pth` 파일로 저장.  
학습된 가중치는 `backend/ml/weights/best_model.pth`에 배치하면 서버 시작 시 자동 로드됨.

```bash
# SKAB 데이터셋 다운로드
git clone https://github.com/waico/SKAB.git backend/ml/data/SKAB

# 로컬 학습 (CPU, 약 5-10분)
cd backend
python ml/train_local.py
```

### 모델 구조
```
Input (batch, 60, 8)
    ↓
LSTM (hidden=128, layers=2)
    ↓
Linear (128 → 128)
    ↓
Sigmoid → 이상 확률
```

---

## 프로젝트 구조

```
factory-ai-guard/
├── docker-compose.yml
├── .env
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 엔트리포인트
│   │   ├── routers/             # API 라우터 (8개)
│   │   ├── services/            # ML, LLM, WebSocket 서비스
│   │   ├── models/              # SQLAlchemy ORM
│   │   └── schemas/             # Pydantic 스키마
│   └── ml/
│       ├── weights/             # 학습된 모델 가중치
│       ├── data/                # SKAB 데이터셋
│       └── train_local.py       # 로컬 학습 스크립트
├── frontend/
│   └── src/
│       ├── pages/               # 6개 페이지
│       ├── components/          # 공통 컴포넌트
│       ├── hooks/               # useWebSocket
│       └── utils/               # API 유틸
├── db/
│   └── init.sql                 # DB 초기화 스크립트
└── nginx/
    └── nginx.conf
```

---

## 성능

SKAB 데이터셋 기준 (valve1 폴더):

| 지표 | 값 |
|------|-----|
| Accuracy | 학습 후 확인 |
| Precision | 학습 후 확인 |
| Recall | 학습 후 확인 |
| F1 Score | 학습 후 확인 |

> 실제 학습 결과는 `/api/model/metrics` 또는 대시보드 모델 성능 페이지에서 확인 가능합니다.

---

## 개발 환경

- **OS**: macOS (Apple Silicon M1)
- **IDE**: Visual Studio Code
- **학습 환경**: Google Colab (T4 GPU)
- **Python**: 3.9.6
- **Node.js**: 20.15.0

---

## 🔗 관련 링크

- [SKAB Dataset](https://github.com/waico/SKAB)
- [FastAPI Docs](https://fastapi.tiangolo.com)
- [Anthropic Claude API](https://docs.anthropic.com)
