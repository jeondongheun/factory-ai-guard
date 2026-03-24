#!/usr/bin/env bash
# =============================================================================
#  ChromaDB 빌드 스크립트 (로컬 실행용)
#  Usage: cd backend && bash ml/rag/chroma_build.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=================================================="
echo "  ChromaDB 빌드 스크립트"
echo "  Backend: $BACKEND_DIR"
echo "=================================================="

# ── 1) 가상환경 활성화 (있으면) ────────────────────────────────
if [ -f "$BACKEND_DIR/../venv/bin/activate" ]; then
    source "$BACKEND_DIR/../venv/bin/activate"
    echo "  ✓ venv 활성화"
elif [ -f "$BACKEND_DIR/venv/bin/activate" ]; then
    source "$BACKEND_DIR/venv/bin/activate"
    echo "  ✓ venv 활성화"
fi

# ── 2) 필수 패키지 설치 ────────────────────────────────────────
echo ""
echo "[ 패키지 설치 확인 ]"
pip install -q chromadb sentence-transformers pandas 2>&1 | grep -E "(Successfully|already|error)" || true

# ── 3) ChromaDB 빌드 실행 ──────────────────────────────────────
echo ""
echo "[ ChromaDB 빌드 실행 ]"
cd "$BACKEND_DIR"
PYTHONPATH="$BACKEND_DIR" python3 ml/rag/chroma_embed.py

echo ""
echo "=================================================="
echo "  ✓ 완료!  backend/ml/data/chroma_db/ 에 저장됨"
echo "=================================================="
