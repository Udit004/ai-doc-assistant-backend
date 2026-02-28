# Backend Setup (FastAPI)

## 1. Create and activate virtual environment

### Windows (PowerShell)
```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS/Linux
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Configure environment
```bash
cp .env.example .env
```

On Windows PowerShell:
```powershell
Copy-Item .env.example .env
```

Set these in `.env`:
- `DATABASE_URL=postgresql+psycopg://...` (Neon/PostgreSQL)
- `GEMINI_API_KEY=...`
- `EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2`

On first run, MiniLM weights are downloaded locally by `sentence-transformers`.

## 4. Run the API
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Base URL: `http://localhost:8000`  
API docs: `http://localhost:8000/docs`

## Auth endpoints
- `POST /api/v1/auth/register` with `{ "email": "...", "password": "..." }`
- `POST /api/v1/auth/login` with `{ "email": "...", "password": "..." }`
- `GET /api/v1/auth/me` with `Authorization: Bearer <token>`

`/api/v1/upload` and `/api/v1/chat` are protected by JWT auth.

Ownership model:
- each uploaded document is linked to `users.id` via `documents.user_id`
- each conversation is linked to `users.id` via `conversations.user_id`
- retrieval only searches chunks from documents owned by the authenticated user

## 5. Run tests
```bash
pytest
```

## 6. Run with Docker
```bash
docker build -t ai-doc-assistant-backend .
docker run --rm -p 8000:8000 ai-doc-assistant-backend
```
