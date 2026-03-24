import { useState, useEffect } from 'react';
import { getDetectionHistory } from '../utils/api';

const SEV_CLASS = {
  Normal: 'badge-normal',
  Low:    'badge-low',
  Medium: 'badge-medium',
  High:   'badge-high',
};

export default function History() {
  const [items, setItems]       = useState([]);
  const [loading, setLoading]   = useState(true);
  const [offset, setOffset]     = useState(0);
  const [severity, setSeverity] = useState('');
  const limit = 20;

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const res = await getDetectionHistory(limit, offset);
      setItems(res.data.items);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchHistory(); }, [offset]);

  const filtered = severity ? items.filter(i => i.severity === severity) : items;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 22, background: 'var(--bg)' }}>

      {/* 헤더 */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 className="page-title">탐지 이력</h2>
          <p className="page-sub">이상 탐지 결과 전체 이력</p>
        </div>
        <select
          value={severity}
          onChange={e => setSeverity(e.target.value)}
          className="input"
          style={{ width: 120 }}
        >
          <option value="">전체</option>
          <option value="High">High</option>
          <option value="Medium">Medium</option>
          <option value="Low">Low</option>
          <option value="Normal">Normal</option>
        </select>
      </div>

      {/* 테이블 */}
      <div className="card" style={{ overflow: 'hidden' }}>
        <table className="tbl">
          <thead>
            <tr>
              <th>ID</th>
              <th>시간</th>
              <th>이상 감지</th>
              <th>확률</th>
              <th>심각도</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={5} style={{ textAlign: 'center', padding: '32px 0', color: 'var(--text-3)' }}>
                  불러오는 중...
                </td>
              </tr>
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan={5} style={{ textAlign: 'center', padding: '32px 0', color: 'var(--text-3)' }}>
                  데이터 없음
                </td>
              </tr>
            ) : filtered.map(item => (
              <tr key={item.id}>
                <td style={{ color: 'var(--text-3)', fontVariantNumeric: 'tabular-nums' }}>
                  #{item.id}
                </td>
                <td style={{ color: 'var(--text-2)' }}>
                  {new Date(item.timestamp).toLocaleString('ko-KR')}
                </td>
                <td>
                  {item.anomaly_detected
                    ? <span style={{ color: 'var(--red)', fontWeight: 600, fontSize: 12 }}>이상</span>
                    : <span style={{ color: 'var(--green)', fontWeight: 600, fontSize: 12 }}>정상</span>
                  }
                </td>
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ width: 64, height: 4, borderRadius: 2, background: 'var(--border)', overflow: 'hidden' }}>
                      <div
                        style={{
                          width: `${item.probability * 100}%`,
                          height: '100%',
                          borderRadius: 2,
                          background: item.probability > 0.7 ? 'var(--red)' : item.probability > 0.4 ? '#EA580C' : 'var(--green)',
                        }}
                      />
                    </div>
                    <span style={{ fontSize: 12, fontVariantNumeric: 'tabular-nums', color: 'var(--text-2)' }}>
                      {(item.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                </td>
                <td>
                  <span className={`badge ${SEV_CLASS[item.severity] ?? 'badge-normal'}`}>
                    {item.severity}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 페이지네이션 */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10 }}>
        <button
          className="btn"
          onClick={() => setOffset(Math.max(0, offset - limit))}
          disabled={offset === 0}
        >
          ← 이전
        </button>
        <span style={{ fontSize: 13, color: 'var(--text-3)', minWidth: 60, textAlign: 'center' }}>
          {offset / limit + 1} 페이지
        </span>
        <button
          className="btn"
          onClick={() => setOffset(offset + limit)}
          disabled={filtered.length < limit}
        >
          다음 →
        </button>
      </div>
    </div>
  );
}
