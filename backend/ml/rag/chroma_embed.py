"""
ChromaDB 임베딩 + RAG 검색 모듈
=================================
역할:
  1) 빌드 모드  : rag_knowledge_base_bom.csv → ChromaDB 임베딩 (직접 실행 시)
  2) 서비스 모드: retrieve_domain_knowledge() / retrieve_for_llm() 제공 (import 시)

변경 사항:
  - 하드코딩된 Mac 경로 → __file__ 기준 상대 경로
  - 신규 카테고리 반영 (detection_strategy, performance_data,
    temperature_detection, vibration_detection, detection_pipeline)
  - retrieve_for_llm(): 탐지 결과 → LLM 프롬프트용 컨텍스트 자동 조합
"""

from __future__ import annotations

import os
import pandas as pd
from pathlib import Path

# ── 경로 설정 (상대 경로, 하드코딩 제거) ──────────────────────────
_BASE        = Path(__file__).parent
CSV_PATH     = _BASE / "rag_knowledge_base_bom.csv"
CHROMA_DIR   = str(_BASE.parent / "data" / "chroma_db")
COLLECTION   = "skab_domain_knowledge"

os.makedirs(CHROMA_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# 문서 텍스트 빌더 (임베딩 품질 핵심)
# ─────────────────────────────────────────────────────────────────

# 카테고리별 한글 레이블
_CAT_LABEL = {
    "operating_mode":       "운영 모드 정상 범위",
    "anomaly_pattern":      "이상 패턴",
    "mode_detection":       "모드 판별",
    "data_quality":         "데이터 품질",
    "vibration_standard":   "진동 표준 (ISO)",
    "temperature_standard": "온도 표준",
    "fault_diagnosis":      "고장 진단",
    "operating_rule":       "운영 규칙",
    "spc_rule":             "SPC 관리 규칙",
    "changepoint_structure":"변화점 구조",
    "sensor_correlation":   "다중 센서 상관",
    "vibration_diagnosis":  "진동 진단 (ISO 13373)",
    "detection_strategy":   "모드별 탐지 전략",
    "performance_data":     "탐지 성능 데이터 (AUC)",
    "temperature_detection":"온도 트렌드 탐지 규칙",
    "vibration_detection":  "진동 DFT 탐지 규칙",
    "detection_pipeline":   "전체 탐지 파이프라인",
}

_MODE_KO = {
    "high_flow":     "고유량 모드 (>100 L/min)",
    "mid_flow":      "중유량 모드 (50~100 L/min)",
    "low_flow":      "저유량 모드 (<50 L/min)",
    "all":           "전체 모드",
    "mid_low_flow":  "중/저유량 모드",
}


def build_document(row: pd.Series) -> str:
    """
    CSV 행 → RAG 검색용 자연어 문서 생성

    검색 품질을 위해:
    - 카테고리/이상유형/센서/모드를 앞에 명시 (토픽 앵커)
    - 임계값/정상범위 수치 포함 (숫자 검색 지원)
    - description을 메인 텍스트로
    - action을 끝에 추가
    """
    parts = []

    # ── 카테고리 레이블 ────────────────────────────────────────
    cat = str(row.get("category", ""))
    parts.append(_CAT_LABEL.get(cat, cat))

    # ── 이상 유형 ──────────────────────────────────────────────
    atype = str(row.get("anomaly_type", "normal"))
    if atype not in ("normal", "summary", "auc_before_mode_separation",
                     "auc_after_mode_separation", "trend_rule",
                     "continuous_decrease", "dft_pipeline", "spc_on_dft",
                     "full_pipeline", "nan"):
        parts.append(f"이상 유형: {atype}")

    # ── 센서 ───────────────────────────────────────────────────
    sensor = str(row.get("sensor", "all"))
    if sensor not in ("all", "nan"):
        parts.append(f"센서: {sensor}")

    # ── 운영 모드 ──────────────────────────────────────────────
    mode = str(row.get("operating_mode", "all"))
    mode_ko = _MODE_KO.get(mode, mode)
    if mode not in ("all", "nan"):
        parts.append(f"운영 모드: {mode_ko}")

    # ── 정상 범위 ──────────────────────────────────────────────
    nr = str(row.get("normal_range", "N/A"))
    if nr not in ("N/A", "", "nan"):
        parts.append(f"정상 범위: {nr}")

    # ── 임계값 ─────────────────────────────────────────────────
    alarm = str(row.get("alarm_threshold", "N/A"))
    trip  = str(row.get("trip_threshold",  "N/A"))
    if alarm not in ("N/A", "", "nan"):
        parts.append(f"알람 임계값: {alarm}")
    if trip not in ("N/A", "", "nan"):
        parts.append(f"트립 임계값: {trip}")

    # ── 핵심 설명 (항상 포함) ──────────────────────────────────
    desc = str(row.get("description", ""))
    if desc and desc != "nan":
        parts.append(desc)

    # ── 조치사항 ───────────────────────────────────────────────
    action = str(row.get("action", ""))
    if action not in ("N/A", "", "nan"):
        parts.append(f"조치: {action}")

    return " | ".join(parts)


def build_metadata(row: pd.Series) -> dict:
    """CSV 행 → ChromaDB 메타데이터 (필터링용)"""
    return {
        "id":              str(row.get("id", "")),
        "source":          str(row.get("source", "")),
        "category":        str(row.get("category", "")),
        "anomaly_type":    str(row.get("anomaly_type", "")),
        "sensor":          str(row.get("sensor", "")),
        "operating_mode":  str(row.get("operating_mode", "")),
        "pattern":         str(row.get("pattern", "")),
        "normal_range":    str(row.get("normal_range", "")),
        "alarm_threshold": str(row.get("alarm_threshold", "")),
        "trip_threshold":  str(row.get("trip_threshold", "")),
        "reference":       str(row.get("reference", "")),
    }


# ─────────────────────────────────────────────────────────────────
# ChromaDB 클라이언트 (lazy 초기화)
# ─────────────────────────────────────────────────────────────────

_client     = None
_collection = None


def _get_collection(read_only: bool = True):
    """ChromaDB 컬렉션 lazy 반환"""
    global _client, _collection

    if _collection is not None:
        return _collection

    import chromadb
    from chromadb.utils import embedding_functions

    _client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb_fn  = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    try:
        _collection = _client.get_collection(
            name=COLLECTION,
            embedding_function=emb_fn,
        )
    except Exception:
        if read_only:
            raise RuntimeError(
                f"ChromaDB 컬렉션 '{COLLECTION}' 없음. "
                "python chroma_embed.py 를 먼저 실행하세요."
            )
        # 빌드 모드: 새 컬렉션 생성
        _collection = _client.create_collection(
            name=COLLECTION,
            embedding_function=emb_fn,
            metadata={"description": "SKAB 워터펌프 도메인 지식 + ISO 표준 + RAAD-LLM"},
        )

    return _collection


# ─────────────────────────────────────────────────────────────────
# RAG 검색 함수
# ─────────────────────────────────────────────────────────────────

def retrieve_domain_knowledge(
    query:          str,
    n_results:      int  = 5,
    anomaly_type:   str  = None,
    sensor:         str  = None,
    operating_mode: str  = None,
    category:       str  = None,
) -> list[dict]:
    """
    RAAD-LLM 파이프라인 범용 RAG 검색

    Args:
        query:          검색 쿼리 (센서값, 패턴 설명 등)
        n_results:      반환 문서 수
        anomaly_type:   이상 유형 필터 (valve1 / rotor_imbalance / ...)
        sensor:         센서명 필터 (Accelerometer1RMS / Volume_Flow_RateRMS / ...)
        operating_mode: 모드 필터 (high_flow / mid_flow / low_flow)
        category:       카테고리 필터 (detection_strategy / anomaly_pattern / ...)

    Returns:
        [{"relevance", "source", "category", "document",
          "alarm_threshold", "normal_range", "trip_threshold"}, ...]
    """
    coll = _get_collection()

    # where 필터 조합
    conditions = []
    if anomaly_type:
        conditions.append({"anomaly_type": {"$in": [anomaly_type, "normal", "summary"]}})
    if sensor:
        conditions.append({"sensor": {"$in": [sensor, "all"]}})
    if operating_mode:
        conditions.append({"operating_mode": {"$in": [operating_mode, "all", "mid_low_flow"]}})
    if category:
        conditions.append({"category": {"$eq": category}})

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    res = coll.query(
        query_texts=[query],
        n_results=min(n_results, coll.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        output.append({
            "relevance":       round(1.0 - float(dist), 4),
            "source":          meta["source"],
            "category":        meta["category"],
            "anomaly_type":    meta["anomaly_type"],
            "sensor":          meta["sensor"],
            "operating_mode":  meta["operating_mode"],
            "document":        doc,
            "normal_range":    meta["normal_range"],
            "alarm_threshold": meta["alarm_threshold"],
            "trip_threshold":  meta["trip_threshold"],
        })

    return output


def retrieve_for_llm(
    mode:          str,
    z_score:       float,
    primary_sensor: str,
    fault_type:    str,
    temp_anomaly:  bool = False,
    n_results:     int  = 5,
) -> dict:
    """
    탐지 결과 → LLM 프롬프트용 RAG 컨텍스트 자동 조합

    ml_service.detect() 결과를 받아 LLM에 전달할 최적 컨텍스트를 구성.

    Args:
        mode:            운영 모드 ('high_flow' | 'mid_flow' | 'low_flow')
        z_score:         1순위 센서 z-score
        primary_sensor:  탐지된 1순위 센서명
        fault_type:      추정 고장 유형
        temp_anomaly:    온도 트렌드 이상 여부
        n_results:       각 쿼리당 검색 결과 수

    Returns:
        {
          "strategy":  모드별 탐지 전략 문서,
          "thresholds": 임계값 정보 문서,
          "fault_info": 고장 유형 관련 문서,
          "temp_info":  온도 이상 관련 문서 (선택),
          "query_summary": 생성된 쿼리 텍스트
        }
    """
    mode_ko = _MODE_KO.get(mode, mode)
    sensor_ko = {
        "Accelerometer1RMS":   "진동",
        "Volume_Flow_RateRMS": "유량",
        "Temperature":         "온도",
    }.get(primary_sensor, primary_sensor)

    # ── 1) 모드별 탐지 전략 ─────────────────────────────────────
    strategy_docs = retrieve_domain_knowledge(
        query=f"{mode_ko} 탐지 전략 임계값",
        n_results=2,
        operating_mode=mode,
        category="detection_strategy",
    )

    # ── 2) 임계값 + 정상 범위 ───────────────────────────────────
    thresh_query = (
        f"{mode_ko} {sensor_ko} z-score {z_score:.2f} 임계값 이상 판정"
    )
    thresh_docs = retrieve_domain_knowledge(
        query=thresh_query,
        n_results=n_results,
        sensor=primary_sensor,
        operating_mode=mode,
    )

    # ── 3) 고장 유형 정보 ────────────────────────────────────────
    fault_map = {
        "rotor_imbalance_suspected": "rotor_imbalance",
        "valve_anomaly_mid":         "valve2",
        "valve_anomaly_low":         "valve1",
        "flow_drop_high":            "rotor_imbalance",
        "temperature_trend_anomaly": "valve1",
    }
    atype = fault_map.get(fault_type, fault_type.split("_")[0] if fault_type else None)

    fault_query = f"{fault_type.replace('_', ' ')} 고장 원인 조치"
    fault_docs = retrieve_domain_knowledge(
        query=fault_query,
        n_results=3,
        anomaly_type=atype,
        operating_mode=mode,
    )

    # ── 4) 온도 이상 정보 (보조) ─────────────────────────────────
    temp_docs = []
    if temp_anomaly:
        temp_docs = retrieve_domain_knowledge(
            query="온도 지속 하락 트렌드 밸브 이상",
            n_results=2,
            sensor="Temperature",
            category="temperature_detection",
        )

    # ── 5) AUC 성능 데이터 ───────────────────────────────────────
    perf_docs = retrieve_domain_knowledge(
        query=f"{mode_ko} 탐지 성능 AUC",
        n_results=1,
        operating_mode=mode,
        category="performance_data",
    )

    query_summary = (
        f"[{mode_ko}] {sensor_ko} z-score={z_score:.3f}, "
        f"fault_type={fault_type}, temp_anomaly={temp_anomaly}"
    )

    return {
        "strategy":      strategy_docs,
        "thresholds":    thresh_docs,
        "fault_info":    fault_docs,
        "temp_info":     temp_docs,
        "performance":   perf_docs,
        "query_summary": query_summary,
    }


# ─────────────────────────────────────────────────────────────────
# 빌드 스크립트 (직접 실행 시)
# ─────────────────────────────────────────────────────────────────

def build_chroma_db() -> None:
    """CSV → ChromaDB 임베딩 빌드"""
    import chromadb
    from chromadb.utils import embedding_functions

    print("=" * 60)
    print("  ChromaDB 빌드 시작")
    print("=" * 60)

    # ── CSV 로딩 ──────────────────────────────────────────────
    print(f"\n[ 1 ] CSV 로딩: {CSV_PATH}")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV 파일 없음: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    print(f"  {len(df)}개 레코드 로딩 완료")
    print(f"  신규 카테고리: {df['category'].unique().tolist()}")

    # ── ChromaDB 초기화 ───────────────────────────────────────
    print(f"\n[ 2 ] ChromaDB 초기화: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 기존 컬렉션 삭제 후 재생성 (전체 재임베딩)
    try:
        client.delete_collection(name=COLLECTION)
        print(f"  기존 컬렉션 삭제: {COLLECTION}")
    except Exception:
        pass

    coll = client.create_collection(
        name=COLLECTION,
        embedding_function=emb_fn,
        metadata={"description": "SKAB 워터펌프 도메인 지식 + RAAD-LLM"},
    )
    print(f"  컬렉션 생성 완료: {COLLECTION}")

    # ── 문서 생성 ─────────────────────────────────────────────
    print(f"\n[ 3 ] 문서 텍스트 생성 ({len(df)}개)...")
    documents = []
    metadatas = []
    ids       = []

    for _, row in df.iterrows():
        documents.append(build_document(row))
        metadatas.append(build_metadata(row))
        ids.append(f"doc_{row['id']}")

    print(f"  예시 (id=1): {documents[0][:120]}...")
    print(f"  예시 (id=55 신규): {documents[54][:120]}...")

    # ── 임베딩 (배치) ─────────────────────────────────────────
    print(f"\n[ 4 ] ChromaDB 임베딩 (배치 크기=10)...")
    BATCH = 10
    total = len(documents)

    for i in range(0, total, BATCH):
        coll.add(
            documents=documents[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH],
            ids=ids[i:i+BATCH],
        )
        print(f"  {min(i+BATCH, total):3d}/{total} 완료")

    print(f"\n  임베딩 완료. 저장 경로: {CHROMA_DIR}")
    print(f"  총 문서 수: {coll.count()}")

    # ── 검색 테스트 ───────────────────────────────────────────
    print(f"\n[ 5 ] 검색 테스트...")
    test_queries = [
        ("밸브 닫힘 시 온도 하락 트렌드",            None,                None),
        ("고유량 진동 z-score 2.56 이상 탐지",        "Accelerometer1RMS", "high_flow"),
        ("저유량 유량 z-score 0.45 임계값",           "Volume_Flow_RateRMS","low_flow"),
        ("모드별 탐지 전략 우선순위",                  None,                None),
        ("온도 SPC 트렌드 연속 하락 이상 탐지 규칙",   "Temperature",       None),
    ]

    for q, sensor, mode in test_queries:
        res = retrieve_domain_knowledge(
            query=q, n_results=2,
            sensor=sensor, operating_mode=mode
        )
        print(f"\n  쿼리: \"{q}\"")
        for r in res:
            print(f"    유사도={r['relevance']:.3f} | {r['category']} | "
                  f"{r['document'][:80]}...")

    # ── retrieve_for_llm 테스트 ───────────────────────────────
    print(f"\n[ 6 ] retrieve_for_llm() 테스트...")

    # 전역 컬렉션 초기화
    global _client, _collection
    _client     = client
    _collection = coll

    ctx = retrieve_for_llm(
        mode="low_flow",
        z_score=0.55,
        primary_sensor="Volume_Flow_RateRMS",
        fault_type="valve_anomaly_low",
        temp_anomaly=True,
        n_results=3,
    )
    print(f"  query_summary: {ctx['query_summary']}")
    print(f"  strategy 문서 수: {len(ctx['strategy'])}")
    print(f"  thresholds 문서 수: {len(ctx['thresholds'])}")
    print(f"  fault_info 문서 수: {len(ctx['fault_info'])}")
    print(f"  temp_info 문서 수: {len(ctx['temp_info'])}")

    print(f"\n{'='*60}")
    print(f"  ChromaDB 빌드 완료 ✓  ({total}개 문서)")
    print(f"{'='*60}")


if __name__ == "__main__":
    build_chroma_db()
