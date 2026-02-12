import { useEffect, useState, useRef } from 'react';
import { MessageSquarePlus } from 'lucide-react';
import { useScriptStore } from '../store/scriptStore';

interface Position {
  x: number;
  y: number;
}

export const SelectionMenu: React.FC = () => {
  const { setSelection } = useScriptStore();
  const [position, setPosition] = useState<Position | null>(null);
  const [selectedText, setSelectedText] = useState<string>('');
  const [selectedRange, setSelectedRange] = useState<{ start: number; end: number } | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMouseUp = (e: MouseEvent) => {
      // Small delay to let selection complete
      setTimeout(() => {
        const selection = window.getSelection();
        if (!selection || selection.isCollapsed || selection.toString().trim() === '') {
          setPosition(null);
          return;
        }

        const text = selection.toString();
        const range = selection.getRangeAt(0);
        
        // Find the line indices
        let startNode: Node | null = range.startContainer;
        let endNode: Node | null = range.endContainer;

        while (startNode && startNode.nodeType !== Node.ELEMENT_NODE) {
          startNode = startNode.parentNode;
        }
        while (endNode && endNode.nodeType !== Node.ELEMENT_NODE) {
          endNode = endNode.parentNode;
        }

        const startEl = (startNode as HTMLElement)?.closest('[data-line-index]');
        const endEl = (endNode as HTMLElement)?.closest('[data-line-index]');

        if (!startEl || !endEl) {
          setPosition(null);
          return;
        }

        const startIndex = parseInt(startEl.getAttribute('data-line-index') || '0', 10);
        const endIndex = parseInt(endEl.getAttribute('data-line-index') || '0', 10);

        // Get position for menu
        const rect = range.getBoundingClientRect();
        setPosition({
          x: rect.left + rect.width / 2,
          y: rect.top - 10
        });
        setSelectedText(text);
        setSelectedRange({ start: Math.min(startIndex, endIndex), end: Math.max(startIndex, endIndex) });
      }, 10);
    };

    const handleMouseDown = (e: MouseEvent) => {
      // Hide menu if clicking outside
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setPosition(null);
      }
    };

    const handleScroll = () => {
      setPosition(null);
    };

    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('mousedown', handleMouseDown);
    document.addEventListener('scroll', handleScroll, true);

    return () => {
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('mousedown', handleMouseDown);
      document.removeEventListener('scroll', handleScroll, true);
    };
  }, []);

  const handleAddToChat = () => {
    if (selectedText && selectedRange) {
      setSelection(selectedText, selectedRange.start, selectedRange.end);
      setPosition(null);
      window.getSelection()?.removeAllRanges();
    }
  };

  if (!position) return null;

  return (
    <div
      ref={menuRef}
      className="fixed z-50 transform -translate-x-1/2 -translate-y-full"
      style={{ left: position.x, top: position.y }}
    >
      <div className="bg-gray-900 text-white rounded-lg shadow-xl flex items-center overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        <button
          onClick={handleAddToChat}
          className="flex items-center gap-2 px-3 py-2 text-sm font-medium hover:bg-gray-800 transition-colors"
        >
          <MessageSquarePlus className="w-4 h-4 text-purple-400" />
          <span>Add to AI</span>
        </button>
      </div>
      {/* Arrow */}
      <div className="absolute left-1/2 -translate-x-1/2 top-full">
        <div className="w-0 h-0 border-l-[6px] border-r-[6px] border-t-[6px] border-l-transparent border-r-transparent border-t-gray-900" />
      </div>
    </div>
  );
};
