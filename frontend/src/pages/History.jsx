import { useState, useEffect } from 'react';
import { getDetectionHistory } from '../utils/api';

const severityBadge = {
  Normal: 'bg-green-900 text-green-400',
  Low:    'bg-yellow-900 text-yellow-400',
  Medium: 'bg-orange-900 text-orange-400',
  High:   'bg-red-900 text-red-400',
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

  const filtered = severity
    ? items.filter(i => i.severity === severity)
    : items;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">탐지 이력</h2>
        <select
          value={severity}
          onChange={e => setSeverity(e.target.value)}
          className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm"
        >
          <option value="">전체</option>
          <option value="High">High</option>
          <option value="Medium">Medium</option>
          <option value="Low">Low</option>
          <option value="Normal">Normal</option>
        </select>
      </div>

      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700 text-slate-400">
              <th className="px-4 py-3 text-left">ID</th>
              <th className="px-4 py-3 text-left">시간</th>
              <th className="px-4 py-3 text-left">이상 감지</th>
              <th className="px-4 py-3 text-left">확률</th>
              <th className="px-4 py-3 text-left">심각도</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-slate-400">
                  로딩 중...
                </td>
              </tr>
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-slate-400">
                  데이터 없음
                </td>
              </tr>
            ) : (
              filtered.map(item => (
                <tr key={item.id} className="border-b border-slate-700 hover:bg-slate-700 transition-colors">
                  <td className="px-4 py-3 text-slate-400">#{item.id}</td>
                  <td className="px-4 py-3">
                    {new Date(item.timestamp).toLocaleString('ko-KR')}
                  </td>
                  <td className="px-4 py-3">
                    {item.anomaly_detected
                      ? <span className="text-red-400">⚠️ 이상</span>
                      : <span className="text-green-400">✅ 정상</span>
                    }
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="w-20 bg-slate-600 rounded-full h-1.5">
                        <div
                          className="bg-blue-500 h-1.5 rounded-full"
                          style={{ width: `${item.probability * 100}%` }}
                        />
                      </div>
                      <span>{(item.probability * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${severityBadge[item.severity]}`}>
                      {item.severity}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* 페이지네이션 */}
      <div className="flex justify-center gap-3">
        <button
          onClick={() => setOffset(Math.max(0, offset - limit))}
          disabled={offset === 0}
          className="px-4 py-2 bg-slate-700 rounded-lg text-sm disabled:opacity-40 hover:bg-slate-600"
        >
          ← 이전
        </button>
        <span className="px-4 py-2 text-sm text-slate-400">
          {offset / limit + 1} 페이지
        </span>
        <button
          onClick={() => setOffset(offset + limit)}
          disabled={filtered.length < limit}
          className="px-4 py-2 bg-slate-700 rounded-lg text-sm disabled:opacity-40 hover:bg-slate-600"
        >
          다음 →
        </button>
      </div>
    </div>
  );
}