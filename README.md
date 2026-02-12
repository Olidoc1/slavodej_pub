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
