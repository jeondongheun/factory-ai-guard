import { useState, useEffect } from 'react';
import { getThresholds, updateThreshold } from '../utils/api';

export default function Settings() {
  const [thresholds, setThresholds] = useState([]);
  const [editing, setEditing]       = useState(null);
  const [saving, setSaving]         = useState(false);
  const [saved, setSaved]           = useState(false);

  useEffect(() => {
    getThresholds()
      .then(r => setThresholds(r.data.thresholds))
      .catch(console.error);
  }, []);

  const handleEdit = (threshold) => {
    setEditing({ ...threshold });
  };

  const handleSave = async () => {
    if (!editing) return;
    setSaving(true);
    try {
      await updateThreshold({
        sensor_name:    editing.sensor_name,
        warning_value:  parseFloat(editing.warning_value),
        critical_value: parseFloat(editing.critical_value),
      });
      setThresholds(prev =>
        prev.map(t => t.sensor_name === editing.sensor_name ? editing : t)
      );
      setEditing(null);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      alert('저장 실패');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">임계값 설정</h2>
        {saved && <span className="text-green-400 text-sm">✅ 저장됨</span>}
      </div>

      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700 text-slate-400">
              <th className="px-4 py-3 text-left">센서</th>
              <th className="px-4 py-3 text-left">경고값</th>
              <th className="px-4 py-3 text-left">위험값</th>
              <th className="px-4 py-3 text-left">최종 수정</th>
              <th className="px-4 py-3 text-left">액션</th>
            </tr>
          </thead>
          <tbody>
            {thresholds.map(t => (
              <tr key={t.sensor_name} className="border-b border-slate-700 hover:bg-slate-700">
                <td className="px-4 py-3 font-medium">{t.sensor_name}</td>
                <td className="px-4 py-3">
                  {editing?.sensor_name === t.sensor_name ? (
                    <input
                      type="number"
                      value={editing.warning_value}
                      onChange={e => setEditing({ ...editing, warning_value: e.target.value })}
                      className="w-24 bg-slate-600 border border-slate-500 rounded px-2 py-1"
                    />
                  ) : (
                    <span className="text-yellow-400">{t.warning_value}</span>
                  )}
                </td>
                <td className="px-4 py-3">
                  {editing?.sensor_name === t.sensor_name ? (
                    <input
                      type="number"
                      value={editing.critical_value}
                      onChange={e => setEditing({ ...editing, critical_value: e.target.value })}
                      className="w-24 bg-slate-600 border border-slate-500 rounded px-2 py-1"
                    />
                  ) : (
                    <span className="text-red-400">{t.critical_value}</span>
                  )}
                </td>
                <td className="px-4 py-3 text-slate-400 text-xs">
                  {new Date(t.updated_at).toLocaleString('ko-KR')}
                </td>
                <td className="px-4 py-3">
                  {editing?.sensor_name === t.sensor_name ? (
                    <div className="flex gap-2">
                      <button
                        onClick={handleSave}
                        disabled={saving}
                        className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs disabled:opacity-50"
                      >
                        저장
                      </button>
                      <button
                        onClick={() => setEditing(null)}
                        className="px-3 py-1 bg-slate-600 hover:bg-slate-500 rounded text-xs"
                      >
                        취소
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => handleEdit(t)}
                      className="px-3 py-1 bg-slate-600 hover:bg-slate-500 rounded text-xs"
                    >
                      수정
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">💡 임계값 가이드</h3>
        <div className="space-y-2 text-sm text-slate-400">
          <p>• <span className="text-yellow-400">경고값</span>: 이 값을 초과하면 노란색으로 표시됩니다</p>
          <p>• <span className="text-red-400">위험값</span>: 이 값을 초과하면 빨간색으로 표시되고 이상으로 판정됩니다</p>
          <p>• 유량/전압은 <span className="text-blue-400">낮을수록</span> 위험합니다 (역방향 임계값)</p>
        </div>
      </div>
    </div>
  );
}