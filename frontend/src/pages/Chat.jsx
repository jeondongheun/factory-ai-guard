import { useState, useRef, useEffect } from 'react';
import { sendChatMessage, clearChatSession } from '../utils/api';

const QUICK = [
  { label: '현재 상태',  text: '현재 펌프 상태 알려줘' },
  { label: '이상 시간',  text: '최근 이상 발생 시간 알려줘' },
  { label: '전체 통계',  text: '전체 탐지 통계 요약해줘' },
  { label: '온도 이상',  text: '온도 이상 원인 알려줘' },
  { label: '진동 기준',  text: '고유량 모드 진동 이상 기준 설명해줘' },
  { label: '고장 유형',  text: '탐지 가능한 고장 유형 알려줘' },
];

function BotAvatar() {
  return (
    <div style={{
      width:32, height:32, borderRadius:'50%', flexShrink:0, marginTop:2,
      background:'linear-gradient(135deg,#F59E0B,#D97706)',
      display:'flex', alignItems:'center', justifyContent:'center',
      boxShadow:'0 2px 6px rgba(217,119,6,.3)',
    }}>
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
      </svg>
    </div>
  );
}

function UserAvatar() {
  return (
    <div style={{
      width:32, height:32, borderRadius:'50%', flexShrink:0, marginTop:2,
      background:'var(--surface-2)', border:'1px solid var(--border)',
      display:'flex', alignItems:'center', justifyContent:'center',
    }}>
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text-3)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
        <circle cx="12" cy="7" r="4"/>
      </svg>
    </div>
  );
}

export default function Chat() {
  const [messages, setMessages] = useState([{
    role: 'assistant',
    content: 'FactoryGuard AI 챗봇입니다.\n펌프 이상 탐지, 원인 분석, 조치 방법 등 무엇이든 물어보세요.',
  }]);
  const [input, setInput]     = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSid]   = useState(null);
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior:'smooth' }); }, [messages]);

  const send = async (text) => {
    const msg = (text ?? input).trim();
    if (!msg || loading) return;
    setInput('');
    setMessages(prev => [...prev, { role:'user', content:msg }]);
    setLoading(true);
    try {
      const res = await sendChatMessage(msg, sessionId);
      const { reply, session_id } = res.data;
      setSid(session_id);
      setMessages(prev => [...prev, { role:'assistant', content:reply }]);
    } catch (e) {
      setMessages(prev => [...prev, {
        role:'assistant',
        content:'오류: ' + (e?.response?.data?.detail || e.message || '알 수 없는 오류'),
      }]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const reset = async () => {
    if (sessionId) { try { await clearChatSession(sessionId); } catch(_) {} }
    setSid(null);
    setMessages([{ role:'assistant', content:'대화가 초기화되었습니다.' }]);
  };

  return (
    <div style={{
      flex:1, display:'flex', flexDirection:'column', minHeight:0, overflow:'hidden',
      background:'var(--bg)',
    }}>

      {/* 헤더 */}
      <div className="animate-fade-up" style={{ flexShrink:0, display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:16 }}>
        <div>
          <h2 className="page-title">AI 챗봇</h2>
          <p className="page-sub">RAAD-LLM + RAG 기반 · 자연어 이상 분석</p>
        </div>
        <button className="btn" onClick={reset} style={{ flexShrink:0, marginTop:4 }}>초기화</button>
      </div>

      {/* 빠른 질문 */}
      <div className="animate-fade-up" style={{ flexShrink:0, display:'flex', flexWrap:'wrap', gap:6, marginBottom:16, animationDelay:'.05s' }}>
        {QUICK.map(q => (
          <button
            key={q.label}
            disabled={loading}
            onClick={() => send(q.text)}
            style={{
              padding:'5px 12px',
              background:'var(--surface)',
              border:'1px solid var(--border)',
              borderRadius:9999,
              color:'var(--text-2)',
              fontSize:12, fontWeight:500,
              cursor: loading ? 'not-allowed' : 'pointer',
              opacity: loading ? .4 : 1,
              transition:'all .12s',
              fontFamily:'var(--font)',
            }}
            onMouseEnter={e => { if(!loading){ e.currentTarget.style.background='var(--surface-2)'; e.currentTarget.style.borderColor='var(--border-hover)'; }}}
            onMouseLeave={e => { e.currentTarget.style.background='var(--surface)'; e.currentTarget.style.borderColor='var(--border)'; }}
          >
            {q.label}
          </button>
        ))}
      </div>

      {/* 메시지 목록 */}
      <div style={{
        flex:1, minHeight:0, overflowY:'auto',
        display:'flex', flexDirection:'column', gap:16,
        paddingRight:4, paddingBottom:4,
        background:'var(--bg)',
      }}>
        {messages.map((m, i) => (
          <div
            key={i}
            className="animate-fade-up"
            style={{
              display:'flex',
              justifyContent: m.role==='user' ? 'flex-end' : 'flex-start',
              alignItems:'flex-start',
              gap:10,
              animationDelay:`${i * 0.04}s`,
            }}
          >
            {m.role === 'assistant' && <BotAvatar />}

            <div style={{
              maxWidth:'72%',
              padding:'11px 15px',
              fontSize:13, lineHeight:1.7,
              whiteSpace:'pre-wrap', wordBreak:'break-word',
              borderRadius: m.role==='user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
              background: m.role==='user'
                ? 'linear-gradient(135deg,var(--accent),var(--accent-hover))'
                : 'var(--surface)',
              border: m.role==='user' ? 'none' : '1px solid var(--border)',
              color: m.role==='user' ? '#fff' : 'var(--text-1)',
              boxShadow: m.role==='user'
                ? '0 2px 10px rgba(217,119,6,.25)'
                : 'var(--shadow-sm)',
            }}>
              {m.content}
            </div>

            {m.role === 'user' && <UserAvatar />}
          </div>
        ))}

        {/* 로딩 */}
        {loading && (
          <div style={{ display:'flex', gap:10, alignItems:'flex-start' }} className="animate-fade-in">
            <BotAvatar />
            <div style={{
              padding:'13px 16px', background:'var(--surface)',
              border:'1px solid var(--border)',
              borderRadius:'16px 16px 16px 4px',
              boxShadow:'var(--shadow-sm)',
            }}>
              <div className="loading-dots"><span/><span/><span/></div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* 입력창 */}
      <div className="animate-fade-up" style={{
        flexShrink:0, display:'flex', gap:8, marginTop:14,
        paddingTop:14, borderTop:'1px solid var(--border)',
        background:'var(--bg)', animationDelay:'.1s',
      }}>
        <textarea
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); } }}
          placeholder="질문을 입력하세요... (Enter 전송 · Shift+Enter 줄바꿈)"
          rows={2}
          disabled={loading}
          className="input"
          style={{ flex:1, resize:'none', lineHeight:1.6 }}
        />
        <button
          onClick={() => send()}
          disabled={!input.trim() || loading}
          className="btn btn-primary"
          style={{ paddingLeft:20, paddingRight:20, alignSelf:'stretch', gap:6 }}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
          전송
        </button>
      </div>

      {sessionId && (
        <p style={{ flexShrink:0, fontSize:10, color:'var(--text-3)', textAlign:'right', marginTop:5, background:'var(--bg)' }}>
          세션 {sessionId}
        </p>
      )}
    </div>
  );
}
