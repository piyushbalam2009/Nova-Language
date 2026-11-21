import { Play } from 'lucide-react';

export type AppView = 'workbench' | 'pipeline' | 'workspace' | 'spec';

interface HeaderProps {
  activeView: AppView;
  onSelectView: (view: AppView) => void;
  onRunProgram: () => void;
  isRunning: boolean;
}

const NAV_ITEMS: Array<{ id: AppView; label: string }> = [
  { id: 'workbench', label: 'Workbench' },
  { id: 'pipeline', label: 'Compiler Pipeline' },
  { id: 'workspace', label: 'Rust Workspace' },
  { id: 'spec', label: 'Specification' },
];

export function Header({ activeView, onSelectView, onRunProgram, isRunning }: HeaderProps) {
  return (
    <header className="h-14 bg-white border-b border-slate-200 px-6 flex items-center justify-between gap-8 shrink-0">
      <button
        type="button"
        onClick={() => onSelectView('workbench')}
        className="font-display text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 rounded"
      >
        Nova
      </button>

      <nav aria-label="Primary navigation" className="flex items-center gap-6 overflow-x-auto">
        {NAV_ITEMS.map((item) => {
          const active = activeView === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelectView(item.id)}
              aria-current={active ? 'page' : undefined}
              className={`h-14 px-1 text-sm font-medium border-b-2 transition-colors whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-blue-600 ${
                active
                  ? 'border-blue-600 text-slate-900'
                  : 'border-transparent text-slate-600 hover:text-slate-900'
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </nav>

      <div className="shrink-0">
        <button
          type="button"
          onClick={onRunProgram}
          disabled={isRunning}
          className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-60 rounded-lg transition-colors whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600"
        >
          <Play className="w-3.5 h-3.5 fill-current" aria-hidden="true" />
          {isRunning ? 'Running...' : 'Run Program'}
        </button>
      </div>
    </header>
  );
}
