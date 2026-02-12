from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import uvicorn
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Import parser services
from services.parser import parse_pdf, parse_fdx

# Load .env from backend directory (works regardless of cwd)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)

app = FastAPI(title="Slavodej API")

# Constants
MAX_PROMPT_LENGTH = 10000  # 10KB max for prompts
MAX_SELECTION_LENGTH = 50000  # 50KB max for selections
MAX_CONTEXT_LENGTH = 100000  # 100KB max for context

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScriptLine(BaseModel):
    type: str  # 'dialogue', 'action', 'heading', 'character', 'parenthetical'
    content: str
    original_text: str

class Scene(BaseModel):
    name: str
    lineIndex: int

class ScriptResponse(BaseModel):
    lines: List[ScriptLine]
    characters: List[str]
    scenes: List[Scene]

class RewriteRequest(BaseModel):
    selection: str = Field(..., min_length=1, max_length=MAX_SELECTION_LENGTH)
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_LENGTH)
    context: Optional[str] = Field(None, max_length=MAX_CONTEXT_LENGTH)
    fileFormat: Optional[str] = Field(None, pattern="^(pdf|fdx)?$")
    
    @field_validator('prompt')
    @classmethod
    def prompt_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Prompt cannot be empty or whitespace only')
        return v.strip()
    
    @field_validator('selection')
    @classmethod
    def selection_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Selection cannot be empty or whitespace only')
        return v

@app.get("/")
async def root():
    return {"message": "Script Slavodej Backend Running"}

@app.post("/upload", response_model=ScriptResponse)
async def upload_script(file: UploadFile = File(...)):
    filename = file.filename
    if not filename or not filename.strip():
        raise HTTPException(status_code=400, detail="No filename provided")
    
    filename_lower = filename.lower().strip()
    
    if not filename_lower.endswith(".pdf") and not filename_lower.endswith(".fdx"):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or FDX.")
    
    try:
        if filename_lower.endswith(".pdf"):
            return await parse_pdf(file)
        else:  # .fdx
            return await parse_fdx(file)
    except ValueError as e:
        # Handle validation errors from parser (file size, corrupted files, etc.)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the actual error but return generic message to avoid leaking info
        print(f"Error processing file: {type(e).__name__}: {e}")
        raise HTTPException(status_code=400, detail="Failed to process file. Please ensure the file is valid.")

@app.post("/rewrite")
async def rewrite_script(request: RewriteRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured. Set GEMINI_API_KEY in .env")

    client = genai.Client(api_key=api_key)
    
    # Build format-specific instructions
    format_info = ""
    if request.fileFormat == "fdx":
        format_info = """
FILE FORMAT: Final Draft (.fdx)
- This is a professional screenplay format
- Character names should be ALL CAPS on their own line
- Dialogue follows immediately after the character name
- Scene headings are ALL CAPS (INT./EXT.)
"""
    elif request.fileFormat == "pdf":
        format_info = """
FILE FORMAT: PDF screenplay
- Standard screenplay formatting applies
- Character names should be ALL CAPS on their own line  
- Dialogue follows immediately after the character name
- Scene headings are ALL CAPS (INT./EXT.)
"""
    
    system_prompt = f"""You are a professional script Slavodej and screenwriter specializing in screenplay formatting.
{format_info}
CRITICAL FORMATTING RULES:
1. PRESERVE the exact screenplay structure. Each element must be on its own line.
2. Scene headings: ALL CAPS (e.g., "INT. COFFEE SHOP - DAY")
3. Character names: ALL CAPS on their own line before dialogue
4. Dialogue: Normal case, on lines following the character name
5. Parentheticals: In parentheses, on their own line between character name and dialogue
6. Action lines: Normal case, full width
7. Match the input structure exactly:
   - If input is ONLY dialogue text, output ONLY dialogue text (no character name)
   - If input includes character name + dialogue, output character name + dialogue
   - Never add screenplay elements that weren't in the original selection

EXAMPLE FORMAT:
INT. OFFICE - NIGHT

SARAH enters, looking exhausted.

SARAH
(sighing)
I can't do this anymore.

JOHN
What do you mean?

OUTPUT RULES:
- Output ONLY the rewritten screenplay text
- Maintain line breaks between different elements
- Do NOT add any commentary, explanations, or markdown
- Do NOT wrap the output in code blocks or quotes
- Match the formatting style of the input exactly
- Keep character names on separate lines from dialogue"""

    user_message = f"""CONTEXT (surrounding script):
{request.context or 'No context provided'}

---

TEXT TO REWRITE:
{request.selection}

---

INSTRUCTION: {request.prompt}

Rewrite the text above following the instruction. Preserve screenplay formatting with proper line breaks."""

    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
            ),
        )
        return {"rewritten_text": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
