# Slavodej

## Prerequisites

- Python 3.12+
- Node.js 18+
- Gemini API key

## Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and replace `your_api_key_here` with your actual Gemini API key.

```bash
python main.py
```

The backend will start at `http://localhost:8000`.

### Frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend will start at `http://localhost:5173`.

### Psycholinguistic Profiling Pipeline

The `db_pipeline.py` script reads works, characters, and dialogues from `slavodej.db` and generates per-character psycholinguistic profile reports (`.md` + `.json`) organized into per-work folders under `psy_profil/output/`.

The pipeline is idempotent -- it skips characters whose `.json` profile already exists on disk. Delete a character's `.json` file to regenerate it.

```bash
cd psy_profil
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The script looks for a Gemini API key (used for AI interpretation) in `psy_profil/.env` or `backend/.env`. Make sure at least one of them contains your key:

```
GEMINI_API_KEY=your_api_key_here
```

Run the pipeline:

```bash
python db_pipeline.py                       # uses default ../slavodej.db
python db_pipeline.py /path/to/slavodej.db  # explicit database path
```

Output is written to `psy_profil/output/<work_title>/<character>.{md,json}`.
