import { useState } from 'react';
import { Sparkles, Send, Trash2, AlertCircle, X } from 'lucide-react';
import { useScriptStore } from '../store/scriptStore';
import axios from 'axios';
import { ScriptMetaPanel } from './ScriptMetaPanel';
import { EditHistory } from './EditHistory';

// API URL - can be overridden with environment variable
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Predefined style presets
const STYLE_PRESETS = [
  { label: 'Tarantino', prompt: 'Rewrite in Quentin Tarantino style - sharp, witty dialogue with pop culture references and tension', color: 'bg-red-500 hover:bg-red-600' },
  { label: 'Sitcom', prompt: 'Rewrite as a sitcom - add humor, comedic timing, and lighthearted banter', color: 'bg-amber-500 hover:bg-amber-600' },
  { label: 'Drama', prompt: 'Make it more dramatic and emotionally intense with deeper subtext', color: 'bg-blue-500 hover:bg-blue-600' },
  { label: 'Noir', prompt: 'Rewrite in film noir style - cynical, moody, with hardboiled dialogue', color: 'bg-gray-700 hover:bg-gray-800' },
  { label: 'Romantic', prompt: 'Add romantic tension and chemistry between the characters', color: 'bg-pink-500 hover:bg-pink-600' },
  { label: 'Thriller', prompt: 'Increase suspense and tension, make it more thrilling', color: 'bg-emerald-600 hover:bg-emerald-700' },
];

export const RightPanel: React.FC = () => {
  const { selection, clearSelection, replaceLines, lines, fileFormat } = useScriptStore();
  const [prompt, setPrompt] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRewrite = async (customPrompt?: string) => {
    const promptToUse = customPrompt || prompt;
    if (!promptToUse.trim() || !selection) return;
    
    setIsProcessing(true);
    setError(null);
    
    try {
      // Get surrounding context (5 lines before and after)
      const startIdx = Math.max(0, selection.startIndex - 5);
      const endIdx = Math.min(lines.length - 1, selection.endIndex + 5);
      const contextLines = lines.slice(startIdx, endIdx + 1).map(l => l.content).join('\n');
      
      const response = await axios.post(`${API_URL}/rewrite`, {
        selection: selection.text,
        prompt: promptToUse,
        context: contextLines,
        fileFormat: fileFormat
      });

      setPreview(response.data.rewritten_text);
    } catch (err) {
      console.error('[RightPanel] Error:', err);
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail;
        if (detail) {
          setError(typeof detail === 'string' ? detail : 'Failed to rewrite. Please try again.');
        } else if (err.code === 'ECONNREFUSED' || err.code === 'ERR_NETWORK') {
          setError('Cannot connect to server. Please check if the backend is running.');
        } else {
          setError('An error occurred. Please try again.');
        }
      } else {
        setError('An unexpected error occurred.');
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleStylePreset = (presetPrompt: string) => {
    setPrompt(presetPrompt);
    handleRewrite(presetPrompt);
  };

  const handleApply = () => {
    if (preview && selection) {
      replaceLines(selection.startIndex, selection.endIndex, preview, prompt || 'Style preset'); 
      clearSelection();
      setPreview(null);
      setPrompt('');
    }
  };

  const handleDiscard = () => {
    setPreview(null);
  };

  const handleClearSelection = () => {
    clearSelection();
    setPreview(null);
    setPrompt('');
    setError(null);
  };

  const dismissError = () => {
    setError(null);
  };

  return (
    <div className="w-80 border-l border-purple-100 bg-gradient-to-b from-white to-purple-50/30 h-full flex flex-col flex-shrink-0 overflow-hidden">
      {/* Characters/Scenes Dropdown */}
      <ScriptMetaPanel />
      
      {/* Edit History */}
      <EditHistory />
      
      {/* AI Rewrite Section */}
      <div className="flex-1 flex flex-col p-4 overflow-hidden">
        <div className="flex items-center gap-2 mb-4">
          <div className="p-1.5 bg-gradient-to-br from-purple-500 to-indigo-500 rounded-lg">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <h2 className="font-semibold text-sm bg-gradient-to-r from-purple-700 to-indigo-600 bg-clip-text text-transparent">AI Rewrite</h2>
        </div>

        {/* Selected Text Display */}
        <div className="flex-shrink-0 mb-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-purple-600 uppercase tracking-wide">Selected Text</span>
            {selection && (
              <button 
                onClick={handleClearSelection}
                className="text-xs text-gray-400 hover:text-red-500 transition-colors"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            )}
          </div>
          <div className="bg-white border border-purple-100 rounded-xl p-3 min-h-[80px] max-h-[120px] overflow-y-auto shadow-sm">
            {selection ? (
              <pre className="text-sm text-gray-700 whitespace-pre-wrap font-mono leading-relaxed">
                {selection.text}
              </pre>
            ) : (
              <p className="text-sm text-gray-400 italic">
                Select text in the script to add it here.
              </p>
            )}
          </div>
          {selection && (
            <p className="text-xs text-purple-400 mt-1">
              Lines {selection.startIndex + 1} - {selection.endIndex + 1}
            </p>
          )}
        </div>

        {/* Style Presets */}
        {selection && !preview && (
          <div className="flex-shrink-0 mb-4">
            <span className="text-xs font-medium text-purple-600 uppercase tracking-wide mb-2 block">Quick Styles</span>
            <div className="flex flex-wrap gap-1.5">
              {STYLE_PRESETS.map((preset) => (
                <button
                  key={preset.label}
                  onClick={() => handleStylePreset(preset.prompt)}
                  disabled={isProcessing}
                  className={`px-2.5 py-1 text-xs font-medium text-white rounded-full ${preset.color} transition-all disabled:opacity-50 shadow-sm`}
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="flex-shrink-0 mb-4 bg-red-50 border border-red-200 rounded-xl p-3 relative">
            <button 
              onClick={dismissError}
              className="absolute top-2 right-2 text-red-400 hover:text-red-600 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
            <div className="flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
              <p className="text-sm text-red-700 pr-4">{error}</p>
            </div>
          </div>
        )}

        {/* Preview or Prompt Input */}
        {preview ? (
          <div className="flex-1 flex flex-col min-h-0">
            <span className="text-xs font-medium text-emerald-600 uppercase tracking-wide mb-2">Preview</span>
            <div className="flex-1 bg-gradient-to-br from-emerald-50 to-teal-50 border border-emerald-200 rounded-xl p-3 overflow-y-auto mb-3 shadow-sm">
              <pre className="text-sm text-gray-800 whitespace-pre-wrap font-mono leading-relaxed">
                {preview}
              </pre>
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleDiscard}
                className="flex-1 py-2.5 text-sm font-medium text-gray-600 hover:text-gray-900 bg-white border border-gray-200 rounded-xl hover:bg-gray-50 transition-colors shadow-sm"
              >
                Discard
              </button>
              <button
                onClick={handleApply}
                className="flex-1 py-2.5 text-sm font-medium text-white bg-gradient-to-r from-emerald-500 to-teal-500 rounded-xl hover:from-emerald-600 hover:to-teal-600 transition-all shadow-sm"
              >
                Apply
              </button>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col min-h-0">
            <span className="text-xs font-medium text-purple-600 uppercase tracking-wide mb-2">Custom Instructions</span>
            <textarea
              className="flex-1 min-h-[80px] p-3 text-sm border border-purple-100 rounded-xl bg-white focus:ring-2 focus:ring-purple-400 focus:border-transparent outline-none resize-none placeholder-gray-400 shadow-sm"
              placeholder="Or describe how to change this..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={!selection}
            />
            <button
              onClick={() => handleRewrite()}
              disabled={!prompt.trim() || !selection || isProcessing}
              className="mt-3 w-full flex items-center justify-center gap-2 bg-gradient-to-r from-purple-600 to-indigo-600 text-white py-2.5 rounded-xl text-sm font-medium hover:from-purple-700 hover:to-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-sm"
            >
              {isProcessing ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  Rewrite
                </>
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

