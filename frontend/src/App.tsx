import { ScriptViewer } from './components/ScriptViewer';
import { RightPanel } from './components/RightPanel';
import { SelectionMenu } from './components/SelectionMenu';
import { FileText, X } from 'lucide-react';
import { useScriptStore } from './store/scriptStore';

function App() {
  const { lines, clearScript, fileFormat } = useScriptStore();
  const hasScript = lines.length > 0;

  return (
    <div className="flex h-screen w-full bg-gradient-to-br from-slate-50 to-purple-50/30 overflow-hidden font-sans text-gray-900">
      {/* Main Content Area */}
      <div className="flex-1 flex flex-col relative min-w-0">
        {/* Colorful Header */}
        <header className="h-14 border-b border-purple-100 flex items-center px-6 bg-white/80 backdrop-blur-sm shrink-0 z-10">
          <div className="flex items-center gap-2 text-gray-900">
             <div className="bg-gradient-to-br from-purple-600 to-indigo-600 text-white p-1.5 rounded-lg shadow-sm">
               <FileText className="w-4 h-4" />
             </div>
             <span className="font-bold tracking-tight text-lg bg-gradient-to-r from-purple-700 to-indigo-600 bg-clip-text text-transparent">Slavodej</span>
          </div>
          
          {/* Script info and close button */}
          {hasScript && (
            <div className="ml-6 flex items-center gap-3">
              <span className="text-xs font-medium text-purple-600 bg-purple-100 px-2 py-1 rounded-full uppercase">
                {fileFormat || 'script'}
              </span>
              <button
                onClick={clearScript}
                className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-red-500 hover:bg-red-50 px-2 py-1 rounded-lg transition-colors"
              >
                <X className="w-3.5 h-3.5" />
                Close script
              </button>
            </div>
          )}
          
          <div className="ml-auto text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded-full">
            v0.1.0-alpha
          </div>
        </header>

        {/* Script Editor/Viewer */}
        <ScriptViewer />
      </div>

      {/* Right Panel - Always visible */}
      <RightPanel />

      {/* Floating Selection Menu */}
      {hasScript && <SelectionMenu />}
    </div>
  );
}

export default App;
