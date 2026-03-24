import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Component } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import History from './pages/History';
import Statistics from './pages/Statistics';
import ModelInfo from './pages/ModelInfo';
import Upload from './pages/Upload';
import Settings from './pages/Settings';
import Chat from './pages/Chat';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '32px' }}>
          <div className="card" style={{ padding: '24px', borderLeft: '3px solid var(--red)' }}>
            <h3 style={{ color: 'var(--red)', marginBottom: 8 }}>렌더링 오류</h3>
            <pre style={{ fontSize: 12, color: 'var(--text-2)', whiteSpace: 'pre-wrap', background: 'var(--surface-2)', padding: 12, borderRadius: 6 }}>
              {this.state.error?.toString()}
            </pre>
            <button className="btn btn-primary" style={{ marginTop: 16 }}
              onClick={() => this.setState({ hasError: false, error: null })}>
              다시 시도
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  return (
    <BrowserRouter>
      <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: 'var(--bg)', fontFamily: 'var(--font)' }}>
        <Sidebar />
        <main style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', background: 'var(--bg)' }}>
          <div style={{ flex: 1, minHeight: 0, position: 'relative' }}>
            <Routes>
              <Route path="/" element={
                <ErrorBoundary><div style={{ position:'absolute',inset:0,overflowY:'auto',padding:'32px 36px',background:'var(--bg)' }}><Dashboard /></div></ErrorBoundary>
              }/>
              <Route path="/history" element={
                <ErrorBoundary><div style={{ position:'absolute',inset:0,overflowY:'auto',padding:'32px 36px',background:'var(--bg)' }}><History /></div></ErrorBoundary>
              }/>
              <Route path="/statistics" element={
                <ErrorBoundary><div style={{ position:'absolute',inset:0,overflowY:'auto',padding:'32px 36px',background:'var(--bg)' }}><Statistics /></div></ErrorBoundary>
              }/>
              <Route path="/model" element={
                <ErrorBoundary><div style={{ position:'absolute',inset:0,overflowY:'auto',padding:'32px 36px',background:'var(--bg)' }}><ModelInfo /></div></ErrorBoundary>
              }/>
              <Route path="/upload" element={
                <ErrorBoundary><div style={{ position:'absolute',inset:0,overflowY:'auto',padding:'32px 36px',background:'var(--bg)' }}><Upload /></div></ErrorBoundary>
              }/>
              <Route path="/settings" element={
                <ErrorBoundary><div style={{ position:'absolute',inset:0,overflowY:'auto',padding:'32px 36px',background:'var(--bg)' }}><Settings /></div></ErrorBoundary>
              }/>
              <Route path="/chat" element={
                <ErrorBoundary><div style={{ position:'absolute',inset:0,display:'flex',flexDirection:'column',padding:'32px 36px',background:'var(--bg)' }}><Chat /></div></ErrorBoundary>
              }/>
            </Routes>
          </div>
        </main>
      </div>
    </BrowserRouter>
  );
}
