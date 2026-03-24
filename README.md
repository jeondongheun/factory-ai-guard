# FactoryGuard

> **스마트팩토리 실시간 이상 탐지 시스템**
> SPC · DFT · Z-score 전처리 → RAG 증강 LLM 진단 → Adaptability(AAD) 자동 베이스라인 업데이트
> FastAPI · React · ChromaDB · Claude API · Docker Compose

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-FF6B35?style=flat-square)

---

## 프로젝트 개요

스마트팩토리 펌프 시스템의 8개 센서 데이터를 실시간으로 수집·분석하여 설비 이상을 자동 탐지하고, RAG 기반 LLM이 원인 분석 및 조치 방안을 제시하는 End-to-End AI 모니터링 시스템.

단순한 임계값 비교를 넘어 **SPC(통계적 공정 관리) → DFT(주파수 분석) → Z-score 정규화** 전처리 파이프라인을 거친 뒤, **ChromaDB RAG + Claude LLM**이 진단하고, 정상 판정 시 **AAD(적응적 베이스라인 업데이트)** 로 기준선이 자동으로 진화하는 구조입니다.

---

## 핵심 파이프라인 — RAAD-LLM

```
Raw Sensor (60-step window)
        │
        ▼
┌───────────────────┐
│  SPC (2-pass MAMR) │  ← 운전 모드별 이상 여부 (Moving Average Moving Range)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  DFT (VibrationPL) │  ← 진동 주파수 스펙트럼 재구성 이상 탐지
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Z-score (다센서) │  ← 센서별 정규화, 이상 스코어 산출
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  RAG (ChromaDB)   │  ← 도메인 지식 벡터 검색 (임계값·고장유형·온도이상)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  LLM (Claude)     │  ← 이상 여부 f(x) ∈ {0,1} + 원인 분석 + 권장 조치
└────────┬──────────┘
         │
    ┌────┴────┐
    ▼         ▼
 f(x)=1     f(x)=0
  이상        정상
  알람         │
              ▼
     ┌─────────────────┐
     │  AAD 베이스라인  │  ← 현재 윈도우 데이터로 SPC/DFT EMA 업데이트
     │  자동 업데이트   │
     └─────────────────┘
```

### 각 단계 설명

| 단계 | 기술 | 역할 |
|------|------|------|
| **SPC** | 2-pass MAMR (Mode-Adaptive Moving Range) | 운전 모드(고유량/저유량)별 이동평균·이동범위 기반 공정 이상 감지 |
| **DFT** | VibrationPipeline FFT | 진동 신호 주파수 스펙트럼을 top-K 주파수로 재구성 후 잔차 이상 탐지 |
| **Z-score** | 센서별 다변량 정규화 | 8개 센서를 각 베이스라인 평균/표준편차로 표준화, 이상 스코어 산출 |
| **RAG** | ChromaDB 벡터 검색 | 운전 모드·Z-score·센서명·고장유형으로 도메인 지식 검색 후 프롬프트 주입 |
| **LLM** | Claude API (Haiku) | RAG 컨텍스트 기반 이진 이상 판정 + 한국어 원인 분석 + 조치 권고 |
| **AAD** | EMA 베이스라인 업데이트 | 정상 판정 시 실제 윈도우 데이터로 SPC/DFT 기준선 점진적 갱신 |

---

## 시스템 아키텍처

```
┌──────────────────────────────────────────────────────┐
│                  React Dashboard (Port 3000)           │
│   실시간 차트 · 센서 카드 · 이상 알람 · AI 챗봇       │
│              Inter 폰트 · 다크 사이드바                │
└──────────────┬────────────────────┬───────────────────┘
               │ REST API           │ WebSocket
               ▼                    ▼
┌──────────────────────────────────────────────────────┐
│               FastAPI Backend (Port 8000)             │
│  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌────────┐ │
│  │ML Service│  │LLM Service│  │ RAG    │  │Sensor  │ │
│  │SPC·DFT  │  │Claude API │  │Chroma  │  │Simulate│ │
│  │Z-score  │  │AAD Update │  │DB      │  │(SKAB)  │ │
│  └──────────┘  └──────────┘  └────────┘  └────────┘ │
└──────────┬────────────────────────────────────────────┘
           │
    ┌──────┴───────┐
    ▼              ▼
PostgreSQL       Redis
(탐지 이력)     (캐시)
```

---

## 주요 기능

### 1. 실시간 대시보드
- 8채널 센서 카드 (진행률 바 · 정상/경고/위험 뱃지)
- WebSocket 기반 실시간 라인차트 (최근 60초)
- 이상 감지 즉시 알람 배너 (심각도별 색상: Normal → High)
- LLM 진단 버튼 → 원인 분석 + 권장 조치 카드 표시

### 2. AI 챗봇
- RAAD-LLM + RAG 기반 자연어 질의응답
- 빠른 질문 버튼 6종 (현재 상태 / 이상 시간 / 통계 / 온도 / 진동 / 고장유형)
- 세션 유지 + 대화 초기화

### 3. CSV 배치 분석
- SKAB 형식 CSV 드래그앤드롭 업로드
- 전체 구간 이상 확률 바차트 시각화
- 검증 결과: **valve1/0.csv** — ground truth 35.0% vs 모델 35.6% (오차 0.6%p)

### 4. 탐지 이력
- 심각도 필터링 · 페이지네이션
- 확률 프로그레스 바 인라인 표시

### 5. 통계
- 일별 이상 발생 추이 바차트
- 정상/이상 비율 도넛 파이차트
- 센서별 평균값 바차트

### 6. 모델 성능
- Accuracy / Precision / Recall / F1 Score
- 레이더 차트 시각화

### 7. 임계값 설정
- 센서별 경고/위험 기준값 커스텀
- DB 저장 및 실시간 반영

---

## REST API 명세

| Method | Endpoint | 설명 |
|--------|----------|------|
| `GET`  | `/api/sensors/current` | 현재 센서값 조회 |
| `GET`  | `/api/sensors/history` | 센서 이력 조회 |
| `WS`   | `/ws/realtime` | 실시간 WebSocket 스트림 |
| `POST` | `/api/detection/analyze` | 이상 탐지 분석 요청 |
| `GET`  | `/api/detection/history` | 탐지 이력 조회 |
| `POST` | `/api/diagnosis/llm` | LLM 상세 진단 요청 |
| `POST` | `/api/chat` | AI 챗봇 메시지 전송 |
| `DELETE` | `/api/chat/{session_id}` | 챗봇 세션 초기화 |
| `GET`  | `/api/stats/summary` | 통계 요약 |
| `GET`  | `/api/stats/trend` | 기간별 트렌드 |
| `GET`  | `/api/model/metrics` | 모델 성능 지표 |
| `POST` | `/api/upload/csv` | CSV 배치 분석 |
| `PUT`  | `/api/settings/thresholds` | 임계값 설정 |

> 전체 API 문서: `http://localhost:8000/docs` (Swagger UI 자동 생성)

---

## 기술 스택

### Backend
| 기술 | 버전 | 용도 |
|------|------|------|
| FastAPI | 0.109 | REST API + WebSocket 서버 |
| SQLAlchemy (async) | 2.x | ORM (PostgreSQL 비동기 연결) |
| PyTorch | 2.2 | LSTM 모델 추론 |
| Anthropic SDK | latest | Claude API (LLM 진단·챗봇) |
| ChromaDB | latest | 도메인 지식 벡터 DB (RAG) |
| Redis | 7 | 실시간 데이터 캐시 |
| NumPy / SciPy | - | SPC · DFT · Z-score 전처리 |

### Frontend
| 기술 | 버전 | 용도 |
|------|------|------|
| React | 18 | UI 프레임워크 |
| Recharts | latest | 센서 데이터 시각화 |
| React Router | 6 | SPA 라우팅 |
| TailwindCSS | 3 | 유틸리티 CSS |
| Axios | latest | REST API 통신 |
| Inter (Google Fonts) | - | 타이포그래피 |

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
- Anthropic API Key

### 1. 레포지토리 클론
```bash
git clone https://github.com/jeondongheun/factory-ai-guard.git
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

### 4. RAG 지식베이스 임베딩
```bash
cd backend
python chroma_embed.py
```

### 5. 백엔드 실행
```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 6. 프론트엔드 실행
```bash
cd frontend
npm install
npm start
```

### 7. 접속
| 서비스 | URL |
|--------|-----|
| 대시보드 | http://localhost:3000 |
| API 문서 | http://localhost:8000/docs |

---

## 모델 학습

모델은 SKAB 데이터셋으로 학습 후 `.pth` 파일로 저장.
학습된 가중치는 `backend/ml/weights/best_model.pth`에 배치하면 서버 시작 시 자동 로드됩니다.

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
LSTM (hidden=128, layers=2, dropout=0.3)
    ↓
Linear (128 → 128) + ReLU
    ↓
Linear (128 → 1) + Sigmoid
    ↓
이상 확률 ∈ [0, 1]
```

---

## 검증 결과

SKAB valve1/0.csv 기준:

| 지표 | 값 |
|------|-----|
| Ground Truth 이상 비율 | 35.0% (401 / 1,147행) |
| 모델 평균 이상 확률 | 35.6% |
| 오차 | **0.6%p** |

> 전체 성능 지표(Accuracy / Precision / Recall / F1)는 `/api/model/metrics` 또는 대시보드 모델 성능 페이지에서 확인 가능합니다.

---

## 프로젝트 구조

```
factory-ai-guard/
├── docker-compose.yml
├── .env
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI 엔트리포인트
│   │   ├── routers/                # API 라우터
│   │   └── services/
│   │       ├── ml_service.py       # SPC · DFT · Z-score · AAD
│   │       ├── llm_service.py      # RAG + Claude 진단·챗봇
│   │       └── websocket_service.py
│   ├── chroma_embed.py             # ChromaDB 도메인 지식 임베딩
│   └── ml/
│       ├── weights/                # 학습된 모델 가중치 (.pth)
│       ├── data/SKAB/              # SKAB 데이터셋
│       └── train_local.py          # 로컬 학습 스크립트
├── frontend/
│   └── src/
│       ├── pages/                  # 7개 페이지
│       ├── components/Sidebar.jsx  # 다크 사이드바
│       ├── hooks/useWebSocket.js   # 실시간 WebSocket 훅
│       └── utils/api.js            # API 유틸
├── db/
│   └── init.sql
└── nginx/
    └── nginx.conf
```

---

## 데이터셋

| 데이터셋 | 설명 | 센서 수 | 출처 |
|---------|------|--------|------|
| **SKAB** | 스마트팩토리 워터 펌프 시스템 이상 탐지 벤치마크 | 8개 (가속도계×2, 온도, 열전대, 압력, 전류, 전압, 유량) | [GitHub](https://github.com/waico/SKAB) |

---

## 개발 환경

- **OS**: macOS (Apple Silicon M1)
- **IDE**: Visual Studio Code
- **Python**: 3.9+
- **Node.js**: 20.x

---

## 관련 링크

- [SKAB Dataset](https://github.com/waico/SKAB)
- [FastAPI Docs](https://fastapi.tiangolo.com)
- [Anthropic Claude API](https://docs.anthropic.com)
- [ChromaDB Docs](https://docs.trychroma.com)
