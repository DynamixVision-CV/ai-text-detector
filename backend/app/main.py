from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .extraction import extract_text, UnsupportedFileType
from .scoring import analyze_document
from .perplexity import perplexity_model_ready
from .models import AnalysisResponse, JobStatus

app = FastAPI(title="AI Text Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB, comfortably covers a ~100-page doc
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt"}

# In-memory job store. Fine for a single-instance deployment; swap for
# Redis/DB if you scale to multiple backend replicas.
_JOBS: dict[str, JobStatus] = {}
_RESULTS: dict[str, AnalysisResponse] = {}


@app.get("/api/health")
def health():
    return {"status": "ok", "perplexity_model_loaded": perplexity_model_ready()}


@app.post("/api/analyze", response_model=JobStatus)
async def analyze(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Format non supporté: {suffix}. Utilisez .pdf, .docx ou .txt")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "Fichier trop volumineux (max 25 Mo)")

    try:
        text = extract_text(file.filename, raw_bytes)
    except UnsupportedFileType as exc:
        raise HTTPException(400, str(exc)) from exc

    if not text.strip():
        raise HTTPException(400, "Aucun texte extrait du document (scan sans OCR ?)")

    job_id = str(uuid.uuid4())
    _JOBS[job_id] = JobStatus(job_id=job_id, status="queued", progress=0.0)

    background_tasks.add_task(_run_analysis, job_id, file.filename, text)

    return _JOBS[job_id]


def _run_analysis(job_id: str, filename: str, text: str) -> None:
    try:
        _JOBS[job_id] = JobStatus(job_id=job_id, status="scoring", progress=0.3,
                                   message="Analyse en cours…")
        result = analyze_document(job_id, filename, text)
        _RESULTS[job_id] = result
        _JOBS[job_id] = JobStatus(job_id=job_id, status="done", progress=1.0)
    except Exception as exc:  # noqa: BLE001
        _JOBS[job_id] = JobStatus(job_id=job_id, status="error", progress=0.0, message=str(exc))


@app.get("/api/status/{job_id}", response_model=JobStatus)
def status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job introuvable")
    return job


@app.get("/api/result/{job_id}", response_model=AnalysisResponse)
def result(job_id: str):
    res = _RESULTS.get(job_id)
    if not res:
        raise HTTPException(404, "Résultat introuvable (job pas encore terminé ?)")
    return res


# --- Serve the static frontend (single-container deployment) ---
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
