import { useState } from 'react';
import { Users, Film, ChevronDown } from 'lucide-react';
import { useScriptStore } from '../store/scriptStore';

type ViewType = 'characters' | 'scenes';

export const ScriptMetaPanel: React.FC = () => {
  const { characters, scenes } = useScriptStore();
  const [activeView, setActiveView] = useState<ViewType>('characters');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  const handleSceneClick = (lineIndex: number) => {
    const element = document.querySelector(`[data-line-index="${lineIndex}"]`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      element.classList.add('bg-yellow-100');
      setTimeout(() => element.classList.remove('bg-yellow-100'), 1500);
    }
  };

  const viewOptions = [
    { value: 'characters' as ViewType, label: 'Characters', icon: Users, count: characters.length, color: 'text-indigo-500', bg: 'bg-indigo-100' },
    { value: 'scenes' as ViewType, label: 'Scenes', icon: Film, count: scenes.length, color: 'text-amber-500', bg: 'bg-amber-100' },
  ];

  const currentOption = viewOptions.find(opt => opt.value === activeView)!;
  const CurrentIcon = currentOption.icon;

  return (
    <div className="border-b border-purple-100 bg-white">
      {/* Dropdown Header */}
      <div className="relative">
        <button
          onClick={() => setIsDropdownOpen(!isDropdownOpen)}
          className="w-full flex items-center gap-2 p-3 text-gray-700 hover:bg-purple-50/50 transition-colors"
        >
          <CurrentIcon className={`w-4 h-4 ${currentOption.color}`} />
          <span className="font-medium text-sm">{currentOption.label}</span>
          <span className={`ml-auto text-xs ${currentOption.color} ${currentOption.bg} px-2 py-0.5 rounded-full font-medium`}>
            {currentOption.count}
          </span>
          <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} />
        </button>

        {/* Dropdown Menu */}
        {isDropdownOpen && (
          <div className="absolute top-full left-0 right-0 z-10 bg-white border border-purple-100 rounded-b-xl shadow-lg overflow-hidden">
            {viewOptions.map((option) => {
              const Icon = option.icon;
              return (
                <button
                  key={option.value}
                  onClick={() => {
                    setActiveView(option.value);
                    setIsDropdownOpen(false);
                  }}
                  className={`w-full flex items-center gap-2 px-3 py-2.5 text-sm transition-colors ${
                    activeView === option.value
                      ? 'bg-purple-50 text-gray-900'
                      : 'text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${option.color}`} />
                  <span>{option.label}</span>
                  <span className={`ml-auto text-xs ${option.color} ${option.bg} px-2 py-0.5 rounded-full font-medium`}>
                    {option.count}
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Content */}
      <div className="px-3 pb-3 space-y-1 max-h-48 overflow-y-auto">
        {activeView === 'characters' && (
          <>
            {characters.length === 0 ? (
              <p className="text-xs text-gray-400 italic pl-2 py-2">No characters detected.</p>
            ) : (
              characters.map((char, idx) => (
                <div 
                  key={idx}
                  className="flex items-center gap-2 pl-2 py-1.5 text-sm text-gray-600 hover:text-indigo-700 hover:bg-indigo-50 rounded-lg transition-colors"
                >
                  <div className="w-6 h-6 rounded-full bg-gradient-to-br from-indigo-400 to-purple-500 flex items-center justify-center text-white text-xs font-medium shadow-sm">
                    {char.charAt(0)}
                  </div>
                  <span className="truncate">{char}</span>
                </div>
              ))
            )}
          </>
        )}

        {activeView === 'scenes' && (
          <>
            {scenes.length === 0 ? (
              <p className="text-xs text-gray-400 italic pl-2 py-2">No scenes detected.</p>
            ) : (
              scenes.map((scene, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSceneClick(scene.lineIndex)}
                  className="w-full flex items-center gap-2 pl-2 py-1.5 text-sm text-gray-600 hover:text-amber-700 hover:bg-amber-50 rounded-lg transition-colors text-left"
                >
                  <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center text-white text-xs font-medium flex-shrink-0 shadow-sm">
                    {idx + 1}
                  </div>
                  <span className="truncate" title={scene.name}>{scene.name}</span>
                </button>
              ))
            )}
          </>
        )}
      </div>
    </div>
  );
};
