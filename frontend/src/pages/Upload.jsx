import { useState } from 'react';
import { uploadCSV, getUploadResult } from '../utils/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function Upload() {
  const [file, setFile]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState(null);
  const [results, setResults] = useState([]);
  const [dragOver, setDragOver] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const res = await uploadCSV(file);
      const { job_id } = res.data;
      const result = await getUploadResult(job_id);
      setSummary(result.data);
      setResults(result.data.results);
    } catch (e) {
      alert('업로드 실패: ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f?.name.endsWith('.csv')) setFile(f);
  };

  const chartData = results
    .filter((_, i) => i % 5 === 0)
    .map(r => ({
      index:       r.index,
      probability: parseFloat((r.probability * 100).toFixed(1)),
    }));

  const anomalyRate = summary ? (summary.anomaly_rate * 100).toFixed(1) : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 22, background: 'var(--bg)' }}>

      {/* 헤더 */}
      <div>
        <h2 style={{ fontSize: 20, fontWeight: 700 }}>CSV 배치 분석</h2>
        <p style={{ color: 'var(--text-3)', fontSize: 13, marginTop: 2 }}>SKAB 형식 데이터 일괄 이상 탐지</p>
      </div>

      {/* 업로드 영역 */}
      <div className="card" style={{ padding: 24 }}>
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          style={{
            border: `2px dashed ${dragOver ? 'var(--accent)' : 'var(--border)'}`,
            borderRadius: 10,
            padding: '40px 24px',
            textAlign: 'center',
            background: dragOver ? 'var(--accent-bg)' : 'var(--surface-2)',
            transition: 'border-color .15s, background .15s',
            cursor: 'default',
          }}
        >
          <div style={{
            width: 44, height: 44, borderRadius: '50%',
            background: 'var(--surface)', border: '1px solid var(--border)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            margin: '0 auto 12px',
          }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--text-3)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/>
              <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
            </svg>
          </div>
          <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-1)', marginBottom: 6 }}>
            {file ? file.name : 'CSV 파일을 끌어다 놓거나 선택하세요'}
          </p>
          <p style={{ fontSize: 12, color: 'var(--text-3)', marginBottom: 16 }}>
            SKAB 형식 · .csv
          </p>
          <input
            type="file"
            accept=".csv"
            onChange={e => setFile(e.target.files[0])}
            style={{ display: 'none' }}
            id="csvInput"
          />
          <label
            htmlFor="csvInput"
            className="btn"
            style={{ cursor: 'pointer', display: 'inline-flex' }}
          >
            파일 선택
          </label>
        </div>

        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="btn btn-primary"
          style={{ marginTop: 16, width: '100%', justifyContent: 'center', padding: '10px 0', fontSize: 14 }}
        >
          {loading ? '분석 중...' : '분석 시작'}
        </button>
      </div>

      {/* 분석 결과 요약 */}
      {summary && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
            <div className="card" style={{ padding: '14px 18px' }}>
              <p className="stat-label">파일명</p>
              <p style={{ fontSize: 14, fontWeight: 600, marginTop: 6, color: 'var(--text-1)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {summary.filename}
              </p>
            </div>
            <div className="card" style={{ padding: '14px 18px' }}>
              <p className="stat-label">총 윈도우</p>
              <p className="stat-value stat-blue">{summary.total_windows}</p>
            </div>
            <div className="card" style={{ padding: '14px 18px' }}>
              <p className="stat-label">이상 감지</p>
              <p className={`stat-value ${parseFloat(anomalyRate) > 20 ? 'stat-high' : 'stat-medium'}`}>
                {summary.anomaly_count}
                <span style={{ fontSize: 13, fontWeight: 400, color: 'var(--text-3)', marginLeft: 6 }}>
                  ({anomalyRate}%)
                </span>
              </p>
            </div>
          </div>

          {/* 이상 확률 차트 */}
          <div className="card" style={{ padding: 20 }}>
            <p style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-2)', marginBottom: 16 }}>
              구간별 이상 확률
            </p>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="index" stroke="var(--border-hover)" tick={{ fontSize: 10, fill: 'var(--text-3)' }} />
                <YAxis stroke="var(--border-hover)" tick={{ fontSize: 10, fill: 'var(--text-3)' }} domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
                  formatter={(v) => [`${v}%`, '이상 확률']}
                  labelStyle={{ color: 'var(--text-2)' }}
                />
                <Bar dataKey="probability" fill="var(--accent)" radius={[3,3,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
