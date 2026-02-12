import { useEffect, useRef } from 'react';
import { useScriptStore } from '../store/scriptStore';
import { Upload } from './Upload';

export const ScriptViewer: React.FC = () => {
  const { lines, setSelection } = useScriptStore();
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
        e.preventDefault();
        
        const selection = window.getSelection();
        if (!selection || selection.isCollapsed) {
          return;
        }

        const range = selection.getRangeAt(0);
        
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

        if (startEl && endEl) {
            const startIndex = parseInt(startEl.getAttribute('data-line-index') || '0', 10);
            const endIndex = parseInt(endEl.getAttribute('data-line-index') || '0', 10);
            const text = selection.toString();

            setSelection(text, Math.min(startIndex, endIndex), Math.max(startIndex, endIndex));
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [setSelection]);

  if (lines.length === 0) {
    return (
        <div className="flex-1 flex flex-col items-center justify-center p-8">
            <Upload />
        </div>
    );
  }

  return (
    <div 
      ref={containerRef}
      className="flex-1 overflow-y-auto bg-white p-12"
    >
      <div className="max-w-3xl mx-auto space-y-1 font-mono text-[14px] leading-[1.6] text-gray-800">
        {lines.map((line, idx) => {
          let className = "relative hover:bg-amber-50/50 transition-colors px-2 py-0.5 rounded select-text";
          
          switch (line.type) {
            case 'heading':
              className += " font-bold uppercase mt-8 mb-2 tracking-wide text-gray-900";
              break;
            case 'character':
              className += " text-center font-bold mt-6 mb-1 uppercase ml-[25%]";
              break;
            case 'dialogue':
              className += " ml-[15%] mr-[15%]";
              break;
            case 'parenthetical':
              className += " ml-[20%] mr-[25%] italic text-gray-600";
              break;
            case 'action':
            default:
              className += " text-left my-2";
              break;
          }

          return (
            <div 
                key={idx} 
                data-line-index={idx}
                className={className}
            >
                <span className="absolute -left-10 top-0.5 text-[10px] text-gray-300 select-none opacity-0 hover:opacity-100 transition-opacity">
                    {idx + 1}
                </span>
                {line.content}
            </div>
          );
        })}
      </div>
      
      <div className="h-48" />
    </div>
  );
};
