import { create } from 'zustand';

export interface ScriptLine {
  type: 'dialogue' | 'action' | 'heading' | 'character' | 'parenthetical';
  content: string;
  original_text: string;
}

export interface Scene {
  name: string;
  lineIndex: number;
}

export type FileFormat = 'pdf' | 'fdx' | null;

// Edit history entry
export interface EditHistoryEntry {
  id: string;
  timestamp: Date;
  startIndex: number;
  endIndex: number;
  originalLines: ScriptLine[];
  newLines: ScriptLine[];
  originalText: string;
  newText: string;
  prompt: string;
}

export interface ScriptState {
  lines: ScriptLine[];
  characters: string[];
  scenes: Scene[];
  fileFormat: FileFormat;
  selection: {
    text: string;
    startIndex: number;
    endIndex: number;
  } | null;
  editHistory: EditHistoryEntry[];
  isLoading: boolean;
  error: string | null;
  setScript: (lines: ScriptLine[], characters: string[], scenes: Scene[], fileFormat: FileFormat) => void;
  clearScript: () => void;
  setSelection: (text: string, startIndex: number, endIndex: number) => void;
  clearSelection: () => void;
  updateLine: (index: number, newContent: string) => void;
  replaceLines: (startIndex: number, endIndex: number, newContent: string, prompt?: string) => void;
  undoEdit: (editId: string) => void;
  clearHistory: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useScriptStore = create<ScriptState>((set) => ({
  lines: [],
  characters: [],
  scenes: [],
  fileFormat: null,
  selection: null,
  editHistory: [],
  isLoading: false,
  error: null,
  setScript: (lines, characters, scenes, fileFormat) => {
    set({ lines, characters, scenes, fileFormat, editHistory: [], error: null });
  },
  clearScript: () => {
    set({ lines: [], characters: [], scenes: [], fileFormat: null, selection: null, editHistory: [], error: null });
  },
  setSelection: (text, startIndex, endIndex) => {
    set({ selection: { text, startIndex, endIndex } });
  },
  clearSelection: () => {
    set({ selection: null });
  },
  updateLine: (index, newContent) => set((state) => {
    const newLines = [...state.lines];
    newLines[index] = { ...newLines[index], content: newContent };
    return { lines: newLines };
  }),
  replaceLines: (startIndex, endIndex, newContent, prompt = 'Manual edit') => set((state) => {
    // Split the new content by lines and create new line objects
    const contentLines = newContent.split('\n').filter(l => l.trim());
    
    // Scene heading pattern: INT., EXT., INT/EXT., I/E., etc.
    const sceneHeadingRe = /^(INT\.|EXT\.|INT\/EXT\.|I\/E\.|INT\.\/EXT\.|EXT\.\/INT\.)/i;
    // Action: stage directions, third-person narrative
    const actionIndicatorRe = /^(REVEAL|CUT TO|ON:|DISSOLVE TO|FADE|PAN TO|[\w\s]+\s+(says|turns|moves|walks|looks|stares|enters|exits)\s)/i;

    const detectType = (content: string, prevType: ScriptLine['type'] | null): ScriptLine['type'] => {
      const trimmed = content.trim();

      // Scene heading
      if (sceneHeadingRe.test(trimmed) && trimmed === trimmed.toUpperCase()) return 'heading';

      // Parenthetical: (direction) - e.g. (full of adrenaline)
      if (trimmed.startsWith('(') && trimmed.endsWith(')')) return 'parenthetical';

      // Character: all caps, short; includes BOYFRIEND (CONT'D)
      if (trimmed === trimmed.toUpperCase() && trimmed.length <= 45 && !trimmed.includes('.')) return 'character';

      // Dialogue: follows character, parenthetical, or dialogue (continued speech)
      if (prevType === 'character' || prevType === 'parenthetical') return 'dialogue';
      if (prevType === 'dialogue') {
        if (actionIndicatorRe.test(trimmed)) return 'action';
        return 'dialogue';
      }

      return 'action';
    };
    
    let prevType: ScriptLine['type'] | null = null;
    const newLineObjects: ScriptLine[] = contentLines.map(content => {
      const type = detectType(content, prevType);
      prevType = type;
      return {
        type,
        content: content.trim(),
        original_text: content
      };
    });
    
    // Store original lines for undo
    const originalLines = state.lines.slice(startIndex, endIndex + 1);
    const originalText = originalLines.map(l => l.content).join('\n');
    
    // Create edit history entry
    const editEntry: EditHistoryEntry = {
      id: `edit-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      startIndex,
      endIndex,
      originalLines: [...originalLines],
      newLines: [...newLineObjects],
      originalText,
      newText: newContent,
      prompt
    };
    
    // Replace the lines from startIndex to endIndex with new lines
    const newLines = [
      ...state.lines.slice(0, startIndex),
      ...newLineObjects,
      ...state.lines.slice(endIndex + 1)
    ];
    
    return { 
      lines: newLines,
      editHistory: [editEntry, ...state.editHistory].slice(0, 50) // Keep last 50 edits
    };
  }),
  undoEdit: (editId) => set((state) => {
    const editIndex = state.editHistory.findIndex(e => e.id === editId);
    if (editIndex === -1) return state;
    
    const edit = state.editHistory[editIndex];
    
    // Find current position of the edited content
    // We need to restore the original lines at the same position
    const newLines = [
      ...state.lines.slice(0, edit.startIndex),
      ...edit.originalLines,
      ...state.lines.slice(edit.startIndex + edit.newLines.length)
    ];
    
    // Remove this edit from history
    const newHistory = state.editHistory.filter(e => e.id !== editId);
    
    return {
      lines: newLines,
      editHistory: newHistory
    };
  }),
  clearHistory: () => set({ editHistory: [] }),
  setLoading: (isLoading) => {
    set({ isLoading });
  },
  setError: (error) => {
    set({ error });
  },
}));
