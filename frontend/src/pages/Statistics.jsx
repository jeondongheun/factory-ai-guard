import { useState, useEffect } from 'react';
import { getStatsSummary, getStatsTrend, getSensorAvg } from '../utils/api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend
} from 'recharts';

const COLORS = ['#22d3ee', '#f97316'];

export default function Statistics() {
  const [summary, setSummary]   = useState(null);
  const [trend, setTrend]       = useState([]);
  const [sensorAvg, setSensorAvg] = useState(null);
  const [days, setDays]         = useState(7);

  useEffect(() => {
    getStatsSummary().then(r => setSummary(r.data)).catch(console.error);
    getSensorAvg().then(r => setSensorAvg(r.data)).catch(console.error);
  }, []);

  useEffect(() => {
    getStatsTrend(days).then(r => setTrend(r.data.trend)).catch(console.error);
  }, [days]);

  const pieData = summary ? [
    { name: '정상', value: summary.total_readings - summary.total_anomalies },
    { name: '이상', value: summary.total_anomalies },
  ] : [];

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">통계</h2>

      {/* 요약 카드 */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: '총 측정',     value: summary.total_readings,  color: 'text-blue-400' },
            { label: '이상 감지',   value: summary.total_anomalies, color: 'text-red-400' },
            { label: '이상 비율',   value: `${(summary.anomaly_rate * 100).toFixed(1)}%`, color: 'text-orange-400' },
            { label: '평균 이상확률', value: `${(summary.avg_probability * 100).toFixed(1)}%`, color: 'text-purple-400' },
          ].map(({ label, value, color }) => (
            <div key={label} className="bg-slate-800 rounded-xl p-4 border border-slate-700">
              <p className="text-xs text-slate-400 mb-1">{label}</p>
              <p className={`text-2xl font-bold ${color}`}>{value}</p>
            </div>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* 트렌드 차트 */}
        <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-slate-300">📈 일별 이상 발생 추이</h3>
            <select
              value={days}
              onChange={e => setDays(Number(e.target.value))}
              className="bg-slate-700 border border-slate-600 rounded px-2 py-1 text-xs"
            >
              <option value={7}>7일</option>
              <option value={14}>14일</option>
              <option value={30}>30일</option>
            </select>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={trend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 10 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
              <Bar dataKey="total"    fill="#334155" name="전체" radius={[4,4,0,0]} />
              <Bar dataKey="anomalies" fill="#f97316" name="이상" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* 파이 차트 */}
        <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">🥧 정상 / 이상 비율</h3>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" outerRadius={80} dataKey="value" label>
                {pieData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
              </Pie>
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 센서 평균값 */}
      {sensorAvg && (
        <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">📊 센서별 평균값</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={Object.entries(sensorAvg).map(([k, v]) => ({ name: k, value: v }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 10 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
              <Bar dataKey="value" fill="#22d3ee" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}