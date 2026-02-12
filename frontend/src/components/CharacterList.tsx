import { useState } from 'react';
import { Users, ChevronDown, ChevronRight } from 'lucide-react';
import { useScriptStore } from '../store/scriptStore';

export const CharacterList: React.FC = () => {
  const { characters } = useScriptStore();
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="border-b border-gray-100">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-2 p-3 text-gray-700 hover:bg-gray-50 transition-colors"
      >
        {isOpen ? (
          <ChevronDown className="w-4 h-4 text-gray-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-400" />
        )}
        <Users className="w-4 h-4" />
        <span className="font-medium text-sm">Characters</span>
        {characters.length > 0 && (
          <span className="ml-auto text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full">
            {characters.length}
          </span>
        )}
      </button>
      
      {isOpen && (
        <div className="px-3 pb-3 space-y-1">
          {characters.length === 0 ? (
            <p className="text-xs text-gray-400 italic pl-6">No characters detected.</p>
          ) : (
            characters.map((char, idx) => (
              <div 
                key={idx}
                className="flex items-center gap-2 pl-6 py-1.5 text-sm text-gray-600 hover:text-gray-900 transition-colors"
              >
                <div className="w-6 h-6 rounded-full bg-gray-200 flex items-center justify-center text-gray-500 text-xs font-medium">
                  {char.charAt(0)}
                </div>
                <span>{char}</span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
};
