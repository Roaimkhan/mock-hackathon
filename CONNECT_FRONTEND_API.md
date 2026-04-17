# Frontend + API Bridge (Without changing Streamlit backend)

This setup keeps your existing Streamlit app untouched in `backend/main.py`.

## What was added

- `api/app.py`: A separate FastAPI bridge exposing `/api/research` and `/api/health`
- `api/requirements.txt`: API dependencies
- `frontend/index.html`, `frontend/styles.css`, `frontend/app.js`: Standalone frontend

## Run steps

1. Start API server

```bash
cd api
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

2. Start frontend (static file server)

```bash
cd frontend
python -m http.server 5500
```

3. Open frontend in browser

- http://127.0.0.1:5500

## Notes

- API reads keys from `backend/.env`, so your current key setup is reused.
- Your Streamlit app can continue running independently on its own port.
- Frontend calls `http://127.0.0.1:8000/api/research`.
