import { useCallback } from 'react';
import { Upload as UploadIcon } from 'lucide-react';
import axios from 'axios';
import { useScriptStore } from '../store/scriptStore';
import type { FileFormat } from '../store/scriptStore';

// API URL - can be overridden with environment variable
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Maximum file size in MB (should match backend)
const MAX_FILE_SIZE_MB = 10;

export const Upload: React.FC = () => {
  const { setScript, setLoading, setError, isLoading } = useScriptStore();

  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Client-side file size validation
    const fileSizeMB = file.size / (1024 * 1024);
    if (fileSizeMB > MAX_FILE_SIZE_MB) {
      setError(`File too large. Maximum size is ${MAX_FILE_SIZE_MB}MB.`);
      return;
    }

    setLoading(true);
    setError(null);

    // Detect file format
    const fileName = file.name.toLowerCase();
    let fileFormat: FileFormat = null;
    if (fileName.endsWith('.pdf')) {
      fileFormat = 'pdf';
    } else if (fileName.endsWith('.fdx')) {
      fileFormat = 'fdx';
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setScript(response.data.lines, response.data.characters, response.data.scenes || [], fileFormat);
    } catch (err) {
      console.error('[Upload] Error:', err);
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail;
        if (detail) {
          setError(typeof detail === 'string' ? detail : 'Failed to process file.');
        } else if (err.code === 'ECONNREFUSED' || err.code === 'ERR_NETWORK') {
          setError('Cannot connect to server. Please check if the backend is running.');
        } else {
          setError('Failed to upload script. Please try again.');
        }
      } else {
        setError('An unexpected error occurred.');
      }
    } finally {
      setLoading(false);
    }
  }, [setScript, setLoading, setError]);

  return (
    <div className="flex flex-col items-center justify-center h-full p-8 text-center border-2 border-dashed border-purple-200 rounded-2xl hover:border-purple-400 transition-all bg-gradient-to-br from-white to-purple-50/50 hover:shadow-lg hover:shadow-purple-100">
      <div className="mb-4 p-4 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-2xl shadow-lg shadow-purple-200">
        {isLoading ? (
           <div className="animate-spin rounded-full h-8 w-8 border-2 border-white/30 border-t-white"></div>
        ) : (
           <UploadIcon className="w-8 h-8 text-white" />
        )}
      </div>
      <h3 className="text-xl font-semibold bg-gradient-to-r from-purple-700 to-indigo-600 bg-clip-text text-transparent mb-2">Upload your script</h3>
      <p className="text-sm text-gray-500 mb-6 max-w-sm">
        Support for PDF (.pdf) and Final Draft (.fdx) files. 
        We'll analyze characters and format automatically.
      </p>
      
      <label className="relative cursor-pointer bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-8 py-3 rounded-xl hover:from-purple-700 hover:to-indigo-700 transition-all font-medium text-sm shadow-lg shadow-purple-200 hover:shadow-xl hover:shadow-purple-300 hover:-translate-y-0.5">
        <span>Select File</span>
        <input 
          type="file" 
          className="hidden" 
          accept=".pdf,.fdx"
          onChange={handleFileUpload}
          disabled={isLoading}
        />
      </label>
      
      <div className="mt-6 flex gap-2">
        <span className="text-xs text-purple-400 bg-purple-50 px-2 py-1 rounded-full">.pdf</span>
        <span className="text-xs text-indigo-400 bg-indigo-50 px-2 py-1 rounded-full">.fdx</span>
      </div>
    </div>
  );
};
