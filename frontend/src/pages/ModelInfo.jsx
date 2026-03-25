import { useState, useEffect } from 'react';
import { getModelMetrics, getModelInfo } from '../utils/api';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer } from 'recharts';

const METRIC_COLOR = {
  Accuracy:  { stat: 'stat-blue',   bar: 'var(--blue)'   },
  Precision: { stat: 'stat-normal', bar: 'var(--green)'  },
  Recall:    { stat: 'stat-medium', bar: 'var(--orange)' },
  'F1 Score':{ stat: 'stat-accent', bar: 'var(--accent)' },
};

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
    <div style={{ display:'flex', flexDirection:'column', gap:24, background:'var(--bg)' }}>

      {/* 헤더 */}
      <div className="animate-fade-up">
        <h2 className="page-title">모델 성능</h2>
        <p className="page-sub">LSTM 이상 탐지 모델 지표</p>
      </div>

      {/* 성능 지표 카드 */}
      {metrics && (
        <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:12 }} className="stagger">
          {[
            { label:'Accuracy',  value: metrics.accuracy  },
            { label:'Precision', value: metrics.precision },
            { label:'Recall',    value: metrics.recall    },
            { label:'F1 Score',  value: metrics.f1        },
          ].map(({ label, value }) => (
            <div key={label} className="card card-hover animate-fade-up" style={{ padding:'16px 18px' }}>
              <p className="stat-label">{label}</p>
              <p className={`stat-value ${METRIC_COLOR[label].stat}`} style={{ fontSize:28 }}>
                {(value * 100).toFixed(1)}
                <span style={{ fontSize:14, fontWeight:400, color:'var(--text-3)' }}>%</span>
              </p>
              <div style={{ marginTop:10 }}>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width:`${value*100}%`, background: METRIC_COLOR[label].bar }} />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:16 }}>

        {/* 레이더 차트 */}
        <div className="card animate-fade-up" style={{ padding:'20px 22px', animationDelay:'.1s' }}>
          <p style={{ fontSize:13, fontWeight:600, color:'var(--text-1)', marginBottom:18 }}>성능 레이더 차트</p>
          <ResponsiveContainer width="100%" height={240}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="var(--border)" />
              <PolarAngleAxis
                dataKey="metric"
                tick={{ fontSize:11, fill:'var(--text-2)', fontFamily:'var(--font)' }}
              />
              <Radar
                dataKey="value"
                stroke="var(--accent)"
                fill="var(--accent)"
                fillOpacity={0.15}
                strokeWidth={2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* 모델 구조 */}
        {info && (
          <div className="card animate-fade-up" style={{ padding:'20px 22px', animationDelay:'.15s' }}>
            <p style={{ fontSize:13, fontWeight:600, color:'var(--text-1)', marginBottom:16 }}>모델 구조</p>
            <div style={{ display:'flex', flexDirection:'column', gap:0 }}>
              {[
                { label:'모델명',      value: info.model_name },
                { label:'입력 차원',   value: info.input_dim },
                { label:'은닉 차원',   value: info.hidden_dim },
                { label:'LSTM 레이어', value: info.num_layers },
                { label:'윈도우 크기', value: info.window_size },
                { label:'디바이스',    value: info.device },
                { label:'모델 상태',   value: info.is_loaded ? '로드됨' : '미로드' },
              ].map(({ label, value }, i, arr) => (
                <div key={label} style={{
                  display:'flex', justifyContent:'space-between', alignItems:'center',
                  padding:'10px 0',
                  borderBottom: i < arr.length-1 ? '1px solid var(--border)' : 'none',
                }}>
                  <span style={{ fontSize:12, color:'var(--text-3)', fontWeight:500 }}>{label}</span>
                  <span style={{
                    fontSize:13, fontWeight:600, color: label==='모델 상태'
                      ? (info.is_loaded ? 'var(--green)' : 'var(--red)')
                      : 'var(--text-1)',
                  }}>
                    {String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
