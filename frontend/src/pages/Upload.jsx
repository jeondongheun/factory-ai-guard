import { useState } from 'react';
import { uploadCSV, getUploadResult } from '../utils/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function Upload() {
  const [file, setFile]         = useState(null);
  const [loading, setLoading]   = useState(false);
  const [summary, setSummary]   = useState(null);
  const [results, setResults]   = useState([]);

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

  // 차트용 데이터 (10개 간격 샘플링)
  const chartData = results
    .filter((_, i) => i % 5 === 0)
    .map(r => ({
      index:       r.index,
      probability: parseFloat((r.probability * 100).toFixed(1)),
      anomaly:     r.anomaly_detected ? 1 : 0,
    }));

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">CSV 배치 분석</h2>

      {/* 업로드 영역 */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <div className="border-2 border-dashed border-slate-600 rounded-xl p-8 text-center">
          <p className="text-4xl mb-3">📁</p>
          <p className="text-slate-300 mb-4">SKAB 형식의 CSV 파일을 업로드하세요</p>
          <input
            type="file"
            accept=".csv"
            onChange={e => setFile(e.target.files[0])}
            className="hidden"
            id="csvInput"
          />
          <label
            htmlFor="csvInput"
            className="cursor-pointer px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm"
          >
            파일 선택
          </label>
          {file && (
            <p className="mt-3 text-sm text-blue-400">선택됨: {file.name}</p>
          )}
        </div>

        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="mt-4 w-full py-3 bg-blue-600 hover:bg-blue-700 rounded-xl font-medium disabled:opacity-40"
        >
          {loading ? '분석 중...' : '🔍 분석 시작'}
        </button>
      </div>

      {/* 분석 결과 요약 */}
      {summary && (
        <>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
              <p className="text-xs text-slate-400 mb-1">파일명</p>
              <p className="font-medium truncate">{summary.filename}</p>
            </div>
            <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
              <p className="text-xs text-slate-400 mb-1">총 윈도우</p>
              <p className="text-2xl font-bold text-blue-400">{summary.total_windows}</p>
            </div>
            <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
              <p className="text-xs text-slate-400 mb-1">이상 감지</p>
              <p className="text-2xl font-bold text-red-400">
                {summary.anomaly_count}
                <span className="text-sm text-slate-400 ml-1">
                  ({(summary.anomaly_rate * 100).toFixed(1)}%)
                </span>
              </p>
            </div>
          </div>

          {/* 이상 확률 차트 */}
          <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
            <h3 className="text-sm font-semibold text-slate-300 mb-4">📈 구간별 이상 확률</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="index" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                  formatter={(v) => [`${v}%`, '이상 확률']}
                />
                <Bar dataKey="probability" fill="#f97316" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}