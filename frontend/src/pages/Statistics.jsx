import { useState, useEffect } from 'react';
import { getStatsSummary, getStatsTrend, getSensorAvg } from '../utils/api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend
} from 'recharts';

const PIE_COLORS = ['#16A34A', '#DC2626'];

export default function Statistics() {
  const [summary, setSummary]     = useState(null);
  const [trend, setTrend]         = useState([]);
  const [sensorAvg, setSensorAvg] = useState(null);
  const [days, setDays]           = useState(7);

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

  const tooltipStyle = {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 8,
    fontSize: 12,
    boxShadow: 'var(--shadow-md)',
  };
  const tickStyle = { fontSize: 10, fill: 'var(--text-3)' };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 22, background: 'var(--bg)' }}>

      {/* 헤더 */}
      <div>
        <h2 style={{ fontSize: 20, fontWeight: 700 }}>통계</h2>
        <p style={{ color: 'var(--text-3)', fontSize: 13, marginTop: 2 }}>탐지 결과 집계 및 센서 평균</p>
      </div>

      {/* 요약 카드 */}
      {summary && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
          {[
            { label: '총 측정',       value: summary.total_readings,                                    cls: 'stat-blue' },
            { label: '이상 감지',     value: summary.total_anomalies,                                   cls: 'stat-high' },
            { label: '이상 비율',     value: `${(summary.anomaly_rate * 100).toFixed(1)}%`,             cls: 'stat-medium' },
            { label: '평균 이상 확률', value: `${(summary.avg_probability * 100).toFixed(1)}%`,         cls: 'stat-neutral' },
          ].map(({ label, value, cls }) => (
            <div key={label} className="card" style={{ padding: '14px 18px' }}>
              <p className="stat-label">{label}</p>
              <p className={`stat-value ${cls}`}>{value}</p>
            </div>
          ))}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

        {/* 트렌드 차트 */}
        <div className="card" style={{ padding: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <p style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-2)' }}>일별 이상 발생 추이</p>
            <select
              value={days}
              onChange={e => setDays(Number(e.target.value))}
              className="input"
              style={{ width: 70, padding: '4px 8px', fontSize: 12 }}
            >
              <option value={7}>7일</option>
              <option value={14}>14일</option>
              <option value={30}>30일</option>
            </select>
          </div>
          <ResponsiveContainer width="100%" height={210}>
            <BarChart data={trend}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="date" stroke="var(--border-hover)" tick={tickStyle} />
              <YAxis stroke="var(--border-hover)" tick={tickStyle} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: 'var(--text-2)' }} />
              <Bar dataKey="total"    fill="var(--border)"  name="전체" radius={[3,3,0,0]} />
              <Bar dataKey="anomalies" fill="var(--accent)" name="이상" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* 파이 차트 */}
        <div className="card" style={{ padding: 20 }}>
          <p style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-2)', marginBottom: 16 }}>
            정상 / 이상 비율
          </p>
          <ResponsiveContainer width="100%" height={210}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%" cy="50%"
                outerRadius={75}
                innerRadius={35}
                dataKey="value"
                paddingAngle={3}
              >
                {pieData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
              </Pie>
              <Tooltip contentStyle={tooltipStyle} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 센서 평균값 */}
      {sensorAvg && (
        <div className="card" style={{ padding: 20 }}>
          <p style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-2)', marginBottom: 16 }}>
            센서별 평균값
          </p>
          <ResponsiveContainer width="100%" height={190}>
            <BarChart data={Object.entries(sensorAvg).map(([k, v]) => ({ name: k, value: v }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" stroke="var(--border-hover)" tick={tickStyle} />
              <YAxis stroke="var(--border-hover)" tick={tickStyle} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: 'var(--text-2)' }} />
              <Bar dataKey="value" fill="var(--blue)" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
