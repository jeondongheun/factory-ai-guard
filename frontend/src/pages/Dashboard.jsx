import { useState } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { getLLMDiagnosis } from '../utils/api';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts';

// 심각도 색상
const severityColor = {
  Normal: 'text-green-400',
  Low:    'text-yellow-400',
  Medium: 'text-orange-400',
  High:   'text-red-400',
};

const severityBg = {
  Normal: 'bg-green-900 border-green-600',
  Low:    'bg-yellow-900 border-yellow-600',
  Medium: 'bg-orange-900 border-orange-600',
  High:   'bg-red-900 border-red-600',
};

// 센서 카드
function SensorCard({ label, value, unit, warning, critical }) {
  const isWarning  = warning  && value >= warning;
  const isCritical = critical && value >= critical;
  const color = isCritical ? 'text-red-400' : isWarning ? 'text-yellow-400' : 'text-green-400';

  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>
        {value?.toFixed(2) ?? '-'}
        <span className="text-sm font-normal text-slate-400 ml-1">{unit}</span>
      </p>
      <p className="text-xs mt-1">
        {isCritical ? '🔴 위험' : isWarning ? '🟡 경고' : '🟢 정상'}
      </p>
    </div>
  );
}

export default function Dashboard() {
  const { sensorData, detection, connected, history } = useWebSocket();
  const [diagnosis, setDiagnosis] = useState(null);
  const [loadingDiag, setLoadingDiag] = useState(false);

  const handleDiagnose = async () => {
    if (!detection || !sensorData) return;
    setLoadingDiag(true);
    try {
      // 탐지 이력에서 최신 ID 가져오기 (간단히 1 사용)
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

  // 차트 데이터 포맷
  const chartData = history.map((d, i) => ({
    t:           i,
    temperature: d.temperature,
    vibration:   d.accelerometer1,
    flow_rate:   d.flow_rate,
    pressure:    d.pressure,
  }));

  return (
    <div className="space-y-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">실시간 대시보드</h2>
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
          <span className="text-sm text-slate-400">{connected ? '실시간 연결됨' : '연결 끊김'}</span>
        </div>
      </div>

      {/* 이상 탐지 상태 배너 */}
      {detection && (
        <div className={`rounded-xl p-4 border ${severityBg[detection.severity] || severityBg.Normal}`}>
          <div className="flex items-center justify-between">
            <div>
              <span className={`text-lg font-bold ${severityColor[detection.severity]}`}>
                {detection.anomaly_detected ? '⚠️ 이상 감지' : '✅ 정상 운전 중'}
              </span>
              <p className="text-sm text-slate-300 mt-1">
                이상 확률: <strong>{(detection.probability * 100).toFixed(1)}%</strong>
                &nbsp;|&nbsp;심각도: <strong>{detection.severity}</strong>
              </p>
            </div>
            {detection.anomaly_detected && (
              <button
                onClick={handleDiagnose}
                disabled={loadingDiag}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium disabled:opacity-50"
              >
                {loadingDiag ? '분석 중...' : '🤖 LLM 진단'}
              </button>
            )}
          </div>
        </div>
      )}

      {/* 센서 카드 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <SensorCard label="온도"     value={sensorData?.temperature}    unit="°C"    warning={30} critical={35} />
        <SensorCard label="진동1"    value={sensorData?.accelerometer1} unit="g"     warning={0.5} critical={1.0} />
        <SensorCard label="유량"     value={sensorData?.flow_rate}      unit="L/min" warning={8} critical={6} />
        <SensorCard label="압력"     value={sensorData?.pressure}       unit="Bar"   warning={2.5} critical={3.0} />
        <SensorCard label="전류"     value={sensorData?.current}        unit="A"     warning={12} critical={15} />
        <SensorCard label="전압"     value={sensorData?.voltage}        unit="V"     />
        <SensorCard label="열전대"   value={sensorData?.thermocouple}   unit="°C"    />
        <SensorCard label="진동2"    value={sensorData?.accelerometer2} unit="g"     warning={0.5} critical={1.0} />
      </div>

      {/* 실시간 차트 */}
      <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">📈 실시간 센서 트렌드 (최근 60초)</h3>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="t" stroke="#64748b" tick={{ fontSize: 11 }} />
            <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            />
            <Legend />
            <Line type="monotone" dataKey="temperature" stroke="#f97316" dot={false} strokeWidth={2} name="온도(°C)" />
            <Line type="monotone" dataKey="vibration"   stroke="#a78bfa" dot={false} strokeWidth={2} name="진동(g)" />
            <Line type="monotone" dataKey="flow_rate"   stroke="#22d3ee" dot={false} strokeWidth={2} name="유량(L/min)" />
            <Line type="monotone" dataKey="pressure"    stroke="#4ade80" dot={false} strokeWidth={2} name="압력(Bar)" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* LLM 진단 결과 */}
      {diagnosis && (
        <div className="bg-slate-800 rounded-xl p-5 border border-blue-700">
          <h3 className="text-sm font-semibold text-blue-400 mb-3">🤖 LLM 진단 결과</h3>
          <div className="space-y-3">
            <div>
              <p className="text-xs text-slate-400 mb-1">원인 분석</p>
              <p className="text-sm text-slate-200">{diagnosis.probable_cause}</p>
            </div>
            <div>
              <p className="text-xs text-slate-400 mb-1">권장 조치</p>
              <p className="text-sm text-slate-200">{diagnosis.recommendation}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}