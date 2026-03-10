import { NavLink } from 'react-router-dom';

const navItems = [
  { path: '/',           icon: '🏭', label: '대시보드' },
  { path: '/history',   icon: '📋', label: '탐지 이력' },
  { path: '/statistics',icon: '📊', label: '통계' },
  { path: '/model',     icon: '🤖', label: '모델 성능' },
  { path: '/upload',    icon: '📁', label: 'CSV 업로드' },
  { path: '/settings',  icon: '⚙️',  label: '설정' },
];

export default function Sidebar() {
  return (
    <aside className="w-56 bg-slate-800 flex flex-col border-r border-slate-700">
      {/* 로고 */}
      <div className="p-5 border-b border-slate-700">
        <h1 className="text-lg font-bold text-blue-400">🏭 FactoryGuard</h1>
        <p className="text-xs text-slate-400 mt-1">AI Anomaly Detection</p>
      </div>

      {/* 네비게이션 */}
      <nav className="flex-1 p-3 space-y-1">
        {navItems.map(({ path, icon, label }) => (
          <NavLink
            key={path}
            to={path}
            end={path === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                isActive
                  ? 'bg-blue-600 text-white font-medium'
                  : 'text-slate-400 hover:bg-slate-700 hover:text-white'
              }`
            }
          >
            <span>{icon}</span>
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* 하단 */}
      <div className="p-4 border-t border-slate-700">
        <p className="text-xs text-slate-500">v1.0.0</p>
      </div>
    </aside>
  );
}