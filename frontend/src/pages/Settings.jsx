import { useState, useEffect } from 'react';
import { getThresholds, updateThreshold } from '../utils/api';

export default function Settings() {
  const [thresholds, setThresholds] = useState([]);
  const [editing, setEditing]       = useState(null);
  const [saving, setSaving]         = useState(false);
  const [saved, setSaved]           = useState(false);

  useEffect(() => {
    getThresholds().then(r => setThresholds(r.data.thresholds)).catch(console.error);
  }, []);

  const handleSave = async () => {
    if (!editing) return;
    setSaving(true);
    try {
      await updateThreshold({
        sensor_name:    editing.sensor_name,
        warning_value:  parseFloat(editing.warning_value),
        critical_value: parseFloat(editing.critical_value),
      });
      setThresholds(prev => prev.map(t => t.sensor_name === editing.sensor_name ? editing : t));
      setEditing(null);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch (e) {
      alert('저장 실패');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:24, background:'var(--bg)' }}>

      {/* 헤더 */}
      <div className="animate-fade-up" style={{ display:'flex', alignItems:'flex-end', justifyContent:'space-between' }}>
        <div>
          <h2 className="page-title">임계값 설정</h2>
          <p className="page-sub">센서별 경고 · 위험 기준값 커스텀</p>
        </div>
        {saved && (
          <span className="badge badge-normal animate-scale-pop" style={{ marginBottom:4 }}>
            저장됨
          </span>
        )}
      </div>

      {/* 테이블 */}
      <div className="card animate-fade-up" style={{ overflow:'hidden', animationDelay:'.05s' }}>
        <table className="tbl">
          <thead>
            <tr>
              <th>센서</th>
              <th>경고값</th>
              <th>위험값</th>
              <th>최종 수정</th>
              <th>액션</th>
            </tr>
          </thead>
          <tbody>
            {thresholds.map(t => {
              const isEditing = editing?.sensor_name === t.sensor_name;
              return (
                <tr key={t.sensor_name}>
                  <td style={{ fontWeight:600 }}>{t.sensor_name}</td>
                  <td>
                    {isEditing ? (
                      <input
                        type="number"
                        value={editing.warning_value}
                        onChange={e => setEditing({ ...editing, warning_value: e.target.value })}
                        className="input"
                        style={{ width:90, padding:'5px 8px' }}
                      />
                    ) : (
                      <span style={{ color:'var(--yellow)', fontWeight:600, fontSize:13 }}>
                        {t.warning_value}
                      </span>
                    )}
                  </td>
                  <td>
                    {isEditing ? (
                      <input
                        type="number"
                        value={editing.critical_value}
                        onChange={e => setEditing({ ...editing, critical_value: e.target.value })}
                        className="input"
                        style={{ width:90, padding:'5px 8px' }}
                      />
                    ) : (
                      <span style={{ color:'var(--red)', fontWeight:600, fontSize:13 }}>
                        {t.critical_value}
                      </span>
                    )}
                  </td>
                  <td style={{ color:'var(--text-3)', fontSize:12 }}>
                    {new Date(t.updated_at).toLocaleString('ko-KR')}
                  </td>
                  <td>
                    {isEditing ? (
                      <div style={{ display:'flex', gap:6 }}>
                        <button className="btn btn-primary" onClick={handleSave} disabled={saving}
                          style={{ padding:'4px 12px', fontSize:12 }}>
                          {saving ? '저장 중...' : '저장'}
                        </button>
                        <button className="btn" onClick={() => setEditing(null)}
                          style={{ padding:'4px 10px', fontSize:12 }}>
                          취소
                        </button>
                      </div>
                    ) : (
                      <button className="btn" onClick={() => setEditing({ ...t })}
                        style={{ padding:'4px 12px', fontSize:12 }}>
                        수정
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* 가이드 */}
      <div className="card animate-fade-up" style={{ padding:'18px 22px', animationDelay:'.1s' }}>
        <p style={{ fontSize:13, fontWeight:600, color:'var(--text-1)', marginBottom:12 }}>임계값 가이드</p>
        <div style={{ display:'flex', flexDirection:'column', gap:8 }}>
          {[
            { color:'var(--yellow)', text:'경고값 초과 시 노란색으로 표시됩니다.' },
            { color:'var(--red)',    text:'위험값 초과 시 빨간색으로 표시되고 이상으로 판정됩니다.' },
            { color:'var(--blue)',   text:'유량·전압은 낮을수록 위험합니다 (역방향 임계값).' },
          ].map(({ color, text }) => (
            <div key={text} style={{ display:'flex', alignItems:'flex-start', gap:10, fontSize:13, color:'var(--text-2)' }}>
              <span style={{ width:8, height:8, borderRadius:'50%', background:color, flexShrink:0, marginTop:4 }} />
              <span>{text}</span>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
