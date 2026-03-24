import { useState } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { getLLMDiagnosis } from '../utils/api';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from 'recharts';

const SEV_BANNER = { Normal:'status-normal', Low:'status-low', Medium:'status-medium', High:'status-high' };
const SEV_BADGE  = { Normal:'badge-normal',  Low:'badge-low',  Medium:'badge-medium',  High:'badge-high'  };
const SEV_STAT   = { Normal:'stat-normal',   Low:'stat-low',   Medium:'stat-medium',   High:'stat-high'   };

/* ── Sensor Card ─────────────────────────────────────────────── */
function SensorCard({ label, value, unit, warning, critical, delay = 0 }) {
  const isCritical = critical != null && value >= critical;
  const isWarning  = warning  != null && value >= warning;
  const statCls    = isCritical ? 'stat-high' : isWarning ? 'stat-medium' : 'stat-normal';
  const badgeCls   = isCritical ? 'badge-high' : isWarning ? 'badge-medium' : 'badge-normal';
  const label2     = isCritical ? '위험' : isWarning ? '경고' : '정상';

  const barPct = value != null && critical != null
    ? Math.min(100, Math.round((value / (critical * 1.2)) * 100))
    : 0;
  const barColor = isCritical ? 'var(--red)' : isWarning ? 'var(--orange)' : 'var(--green)';

  return (
    <div
      className="card card-hover animate-fade-up"
      style={{ padding:'16px 18px', animationDelay:`${delay}s`, cursor:'default' }}
    >
      <p className="stat-label">{label}</p>
      <p className={`stat-value ${statCls}`} style={{ fontSize:24, marginTop:4 }}>
        {value?.toFixed(2) ?? '—'}
        <span style={{ fontSize:11, fontWeight:400, color:'var(--text-3)', marginLeft:4 }}>{unit}</span>
      </p>
      {critical != null && (
        <div style={{ marginTop:8 }}>
          <div className="progress-track">
            <div className="progress-fill" style={{ width:`${barPct}%`, background: barColor }} />
          </div>
        </div>
      )}
      <div style={{ marginTop: critical != null ? 8 : 10, display:'flex', alignItems:'center', justifyContent:'space-between' }}>
        <span className={`badge ${badgeCls}`}>{label2}</span>
        {critical != null && (
          <span style={{ fontSize:10, color:'var(--text-3)' }}>한계 {critical}{unit}</span>
        )}
      </div>
    </div>
  );
}

/* ── Custom Chart Tooltip ────────────────────────────────────── */
function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background:'var(--surface)', border:'1px solid var(--border)',
      borderRadius: 10, padding:'10px 14px', boxShadow:'var(--shadow-md)',
      fontSize:12, fontFamily:'var(--font)',
    }}>
      <p style={{ color:'var(--text-3)', marginBottom:6, fontSize:11 }}>t = {label}</p>
      {payload.map(p => (
        <div key={p.name} style={{ display:'flex', alignItems:'center', gap:6, marginBottom:3 }}>
          <span style={{ width:8, height:8, borderRadius:'50%', background:p.color, flexShrink:0 }}/>
          <span style={{ color:'var(--text-2)' }}>{p.name}</span>
          <span style={{ marginLeft:'auto', fontWeight:600, color:'var(--text-1)' }}>{p.value?.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

export default function Dashboard() {
  const { sensorData, detection, connected, history } = useWebSocket();
  const [diagnosis, setDiagnosis]     = useState(null);
  const [loadingDiag, setLoadingDiag] = useState(false);

  const handleDiagnose = async () => {
    if (!detection || !sensorData) return;
    setLoadingDiag(true);
    try {
      const res = await getLLMDiagnosis(1, {
        temperature:    sensorData.temperature,
        accelerometer1: sensorData.accelerometer1,
        flow_rate:      sensorData.flow_rate,
        pressure:       sensorData.pressure,
        current:        sensorData.current,
      });
      setDiagnosis(res.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingDiag(false);
    }
  };

  const chartData = history.map((d, i) => ({
    t: i, temperature: d.temperature,
    vibration: d.accelerometer1, flow_rate: d.flow_rate, pressure: d.pressure,
  }));

  const sev = detection?.severity ?? 'Normal';

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:24 }}>

      {/* ── 헤더 ── */}
      <div className="animate-fade-up" style={{ display:'flex', alignItems:'flex-end', justifyContent:'space-between' }}>
        <div>
          <h2 className="page-title">실시간 대시보드</h2>
          <p className="page-sub">펌프 센서 실시간 모니터링 · RAAD-LLM</p>
        </div>
        <div style={{ display:'flex', alignItems:'center', gap:8, paddingBottom:4 }}>
          <span className={connected ? 'dot-live' : 'dot-dead'} />
          <span style={{ fontSize:12, color:'var(--text-3)', fontWeight:500 }}>
            {connected ? 'LIVE' : '연결 끊김'}
          </span>
        </div>
      </div>

      {/* ── 이상 탐지 상태 배너 ── */}
      {detection && (
        <div className={`status-banner animate-fade-up ${SEV_BANNER[sev]}`} style={{ animationDelay:'.05s' }}>
          <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', gap:16 }}>
            <div>
              <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:5 }}>
                <span style={{ fontSize:16, fontWeight:700, color:'var(--text-1)' }}>
                  {detection.anomaly_detected ? '이상 감지됨' : '정상 운전 중'}
                </span>
                <span className={`badge ${SEV_BADGE[sev]}`}>{sev}</span>
              </div>
              <div style={{ display:'flex', alignItems:'center', gap:16, fontSize:13, color:'var(--text-2)' }}>
                <span>이상 확률 <strong style={{ color:'var(--text-1)' }}>{(detection.probability * 100).toFixed(1)}%</strong></span>
                {detection.mode && <span style={{ color:'var(--text-3)' }}>모드: {detection.mode}</span>}
                {detection.fault_type && detection.fault_type !== 'unknown' && (
                  <span style={{ color:'var(--text-3)' }}>유형: {detection.fault_type}</span>
                )}
              </div>
            </div>
            {detection.anomaly_detected && (
              <button className="btn btn-primary" onClick={handleDiagnose} disabled={loadingDiag} style={{ flexShrink:0 }}>
                {loadingDiag
                  ? <><span style={{ width:12,height:12,border:'2px solid rgba(255,255,255,.3)',borderTopColor:'white',borderRadius:'50%',display:'inline-block',animation:'spin .7s linear infinite' }}/> 분석 중</>
                  : 'LLM 진단'
                }
              </button>
            )}
          </div>
        </div>
      )}

      {/* ── 센서 카드 그리드 ── */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:12 }} className="stagger">
        <SensorCard label="온도"   value={sensorData?.temperature}    unit="°C"    warning={30}  critical={35}  delay={.06} />
        <SensorCard label="진동 1" value={sensorData?.accelerometer1} unit="g"     warning={0.5} critical={1.0} delay={.09} />
        <SensorCard label="유량"   value={sensorData?.flow_rate}      unit="L/min" warning={8}   critical={6}   delay={.12} />
        <SensorCard label="압력"   value={sensorData?.pressure}       unit="Bar"   warning={2.5} critical={3.0} delay={.15} />
        <SensorCard label="전류"   value={sensorData?.current}        unit="A"     warning={12}  critical={15}  delay={.18} />
        <SensorCard label="전압"   value={sensorData?.voltage}        unit="V"                                  delay={.21} />
        <SensorCard label="열전대" value={sensorData?.thermocouple}   unit="°C"                                 delay={.24} />
        <SensorCard label="진동 2" value={sensorData?.accelerometer2} unit="g"     warning={0.5} critical={1.0} delay={.27} />
      </div>

      {/* ── 실시간 차트 ── */}
      <div className="card animate-fade-up" style={{ padding:'20px 22px', animationDelay:'.3s' }}>
        <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:18 }}>
          <p style={{ fontSize:13, fontWeight:600, color:'var(--text-1)' }}>실시간 센서 트렌드</p>
          <span style={{ fontSize:11, color:'var(--text-3)' }}>최근 60초</span>
        </div>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={chartData} margin={{ top:4, right:8, bottom:0, left:-10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
            <XAxis dataKey="t" stroke="transparent" tick={{ fontSize:10, fill:'var(--text-3)' }} axisLine={false} />
            <YAxis stroke="transparent" tick={{ fontSize:10, fill:'var(--text-3)' }} axisLine={false} tickLine={false} />
            <Tooltip content={<ChartTip />} />
            <Legend iconType="circle" iconSize={7} wrapperStyle={{ fontSize:11, paddingTop:8 }} />
            <Line type="monotone" dataKey="temperature" stroke="#F97316" dot={false} strokeWidth={2} name="온도(°C)" />
            <Line type="monotone" dataKey="vibration"   stroke="#8B5CF6" dot={false} strokeWidth={2} name="진동(g)" />
            <Line type="monotone" dataKey="flow_rate"   stroke="#0EA5E9" dot={false} strokeWidth={2} name="유량(L/min)" />
            <Line type="monotone" dataKey="pressure"    stroke="#10B981" dot={false} strokeWidth={2} name="압력(Bar)" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ── LLM 진단 결과 ── */}
      {diagnosis && (
        <div className="card animate-scale-pop" style={{ padding:'20px 22px', borderLeft:'3px solid var(--accent)' }}>
          <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:16 }}>
            <div style={{ width:24,height:24,borderRadius:6,background:'var(--accent-light)',display:'flex',alignItems:'center',justifyContent:'center' }}>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
              </svg>
            </div>
            <p style={{ fontSize:13, fontWeight:700, color:'var(--accent)' }}>LLM 진단 결과</p>
          </div>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12 }}>
            {[
              { title:'원인 분석', text: diagnosis.probable_cause },
              { title:'권장 조치', text: diagnosis.recommendation },
            ].map(({ title, text }) => (
              <div key={title} style={{ background:'var(--surface-2)', borderRadius:8, padding:'12px 14px' }}>
                <p className="section-label" style={{ marginBottom:6 }}>{title}</p>
                <p style={{ fontSize:13, color:'var(--text-1)', lineHeight:1.65 }}>{text}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
