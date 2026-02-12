import { useState } from 'react';
import { History, Undo2, ChevronDown, ChevronRight, Trash2, Eye } from 'lucide-react';
import { useScriptStore } from '../store/scriptStore';

export const EditHistory: React.FC = () => {
  const { editHistory, undoEdit, clearHistory } = useScriptStore();
  const [isOpen, setIsOpen] = useState(false);
  const [expandedEdit, setExpandedEdit] = useState<string | null>(null);

  if (editHistory.length === 0) {
    return null;
  }

  const formatTime = (date: Date) => {
    return new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const truncateText = (text: string, maxLength: number = 50) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  return (
    <div className="border-b border-purple-100 bg-white">
      {/* Header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-2 p-3 text-gray-700 hover:bg-purple-50/50 transition-colors"
      >
        {isOpen ? (
          <ChevronDown className="w-4 h-4 text-gray-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-400" />
        )}
        <History className="w-4 h-4 text-orange-500" />
        <span className="font-medium text-sm">Edit History</span>
        <span className="ml-auto text-xs text-orange-500 bg-orange-100 px-2 py-0.5 rounded-full font-medium">
          {editHistory.length}
        </span>
      </button>

      {/* Content */}
      {isOpen && (
        <div className="px-3 pb-3">
          {/* Clear all button */}
          <div className="flex justify-end mb-2">
            <button
              onClick={clearHistory}
              className="text-xs text-gray-400 hover:text-red-500 transition-colors flex items-center gap-1"
            >
              <Trash2 className="w-3 h-3" />
              Clear all
            </button>
          </div>

          {/* Edit list */}
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {editHistory.map((edit) => (
              <div
                key={edit.id}
                className="bg-gradient-to-br from-orange-50 to-amber-50 border border-orange-100 rounded-lg p-2 text-xs"
              >
                {/* Edit header */}
                <div className="flex items-center justify-between mb-1">
                  <span className="text-orange-600 font-medium">
                    Lines {edit.startIndex + 1}-{edit.endIndex + 1}
                  </span>
                  <span className="text-gray-400">{formatTime(edit.timestamp)}</span>
                </div>

                {/* Prompt used */}
                <p className="text-gray-600 mb-2 italic">
                  "{truncateText(edit.prompt, 40)}"
                </p>

                {/* Toggle details */}
                <button
                  onClick={() => setExpandedEdit(expandedEdit === edit.id ? null : edit.id)}
                  className="flex items-center gap-1 text-gray-500 hover:text-gray-700 transition-colors mb-2"
                >
                  <Eye className="w-3 h-3" />
                  {expandedEdit === edit.id ? 'Hide changes' : 'View changes'}
                </button>

                {/* Expanded view */}
                {expandedEdit === edit.id && (
                  <div className="space-y-2 mt-2 border-t border-orange-200 pt-2">
                    {/* Original */}
                    <div>
                      <span className="text-red-500 font-medium block mb-1">Original:</span>
                      <div className="bg-red-50 border border-red-200 rounded p-2 font-mono text-gray-700 whitespace-pre-wrap max-h-24 overflow-y-auto">
                        {edit.originalText || '(empty)'}
                      </div>
                    </div>

                    {/* New */}
                    <div>
                      <span className="text-green-600 font-medium block mb-1">Changed to:</span>
                      <div className="bg-green-50 border border-green-200 rounded p-2 font-mono text-gray-700 whitespace-pre-wrap max-h-24 overflow-y-auto">
                        {edit.newText || '(empty)'}
                      </div>
                    </div>
                  </div>
                )}

                {/* Undo button */}
                <button
                  onClick={() => undoEdit(edit.id)}
                  className="w-full mt-2 flex items-center justify-center gap-1 py-1.5 text-orange-600 hover:text-orange-700 bg-white border border-orange-200 rounded-lg hover:bg-orange-50 transition-colors font-medium"
                >
                  <Undo2 className="w-3 h-3" />
                  Undo this edit
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
