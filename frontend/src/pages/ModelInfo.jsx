import { useState, useEffect } from 'react';
import { getModelMetrics, getModelInfo } from '../utils/api';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer } from 'recharts';

export default function ModelInfo() {
  const [metrics, setMetrics] = useState(null);
  const [info, setInfo]       = useState(null);

  useEffect(() => {
    getModelMetrics().then(r => setMetrics(r.data)).catch(console.error);
    getModelInfo().then(r => setInfo(r.data)).catch(console.error);
  }, []);

  const radarData = metrics ? [
    { metric: 'Accuracy',  value: metrics.accuracy  * 100 },
    { metric: 'Precision', value: metrics.precision * 100 },
    { metric: 'Recall',    value: metrics.recall    * 100 },
    { metric: 'F1 Score',  value: metrics.f1        * 100 },
  ] : [];

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">모델 성능</h2>

      {/* 성능 지표 카드 */}
      {metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'Accuracy',  value: metrics.accuracy,  color: 'text-blue-400' },
            { label: 'Precision', value: metrics.precision, color: 'text-green-400' },
            { label: 'Recall',    value: metrics.recall,    color: 'text-orange-400' },
            { label: 'F1 Score',  value: metrics.f1,        color: 'text-purple-400' },
          ].map(({ label, value, color }) => (
            <div key={label} className="bg-slate-800 rounded-xl p-4 border border-slate-700">
              <p className="text-xs text-slate-400 mb-1">{label}</p>
              <p className={`text-2xl font-bold ${color}`}>
                {(value * 100).toFixed(1)}%
              </p>
              {/* 프로그레스 바 */}
              <div className="mt-2 w-full bg-slate-600 rounded-full h-1.5">
                <div
                  className="bg-blue-500 h-1.5 rounded-full"
                  style={{ width: `${value * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* 레이더 차트 */}
        <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">🕸️ 성능 레이더 차트</h3>
          <ResponsiveContainer width="100%" height={250}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#334155" />
              <PolarAngleAxis dataKey="metric" stroke="#64748b" tick={{ fontSize: 12 }} />
              <Radar dataKey="value" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.3} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* 모델 정보 */}
        {info && (
          <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
            <h3 className="text-sm font-semibold text-slate-300 mb-4">🤖 모델 구조</h3>
            <div className="space-y-3">
              {[
                { label: '모델명',      value: info.model_name },
                { label: '입력 차원',   value: info.input_dim },
                { label: '은닉 차원',   value: info.hidden_dim },
                { label: 'LSTM 레이어', value: info.num_layers },
                { label: '윈도우 크기', value: info.window_size },
                { label: '디바이스',    value: info.device },
                { label: '모델 상태',   value: info.is_loaded ? '✅ 로드됨' : '⚠️ 미로드' },
              ].map(({ label, value }) => (
                <div key={label} className="flex justify-between items-center py-2 border-b border-slate-700">
                  <span className="text-sm text-slate-400">{label}</span>
                  <span className="text-sm font-medium">{value}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}