"""FastAPI server for local dataset, training, and inference workflows."""

import json
import os
import hmac
import csv
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from text_classifier.data import read_examples, split_examples, write_examples


ROOT = Path(__file__).resolve().parents[1]
STORAGE = Path(os.environ.get("TRANSFORMER_STORAGE_DIR", str(ROOT / "web_data"))).resolve()
DATASETS = STORAGE / "datasets"
RUNS = STORAGE / "runs"
STATIC = Path(__file__).resolve().parent / "static"
SAMPLE = ROOT / "examples" / "sentiment.csv"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
LOCK = threading.Lock()
JOBS: dict[str, dict] = {}

app = FastAPI(title="Transformer Lab", version="1.0")
API_KEY = os.environ.get("TRANSFORMER_API_KEY", "")
if os.environ.get("TRANSFORMER_REQUIRE_API_KEY") == "1" and not API_KEY:
    raise RuntimeError("TRANSFORMER_API_KEY is required for this deployment")


@app.middleware("http")
async def require_api_key(request, call_next):
    if API_KEY and request.url.path.startswith("/api/") and request.method != "OPTIONS":
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied, f"Bearer {API_KEY}"):
            return JSONResponse({"detail": "An access key is required"}, status_code=401)
    return await call_next(request)


allowed_origins = [origin.strip().rstrip("/") for origin in os.environ.get("TRANSFORMER_ALLOWED_ORIGINS", "").split(",") if origin.strip()]
if allowed_origins:
    app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Authorization"])
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class TrainRequest(BaseModel):
    dataset_id: str
    model_name: str = "google-bert/bert-base-uncased"
    epochs: int = Field(default=3, ge=1, le=20)
    batch_size: int = Field(default=16, ge=1, le=128)
    max_length: int = Field(default=256, ge=8, le=512)
    learning_rate: float = Field(default=2e-5, gt=0, le=0.01)
    test_fraction: float = Field(default=0.15, gt=0, lt=0.5)
    validation_fraction: float = Field(default=0.15, gt=0, lt=0.5)
    seed: int = 42
    device: str = "auto"


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dataset_path(dataset_id: str) -> Path:
    if dataset_id == "sample":
        return SAMPLE
    if not dataset_id.startswith("ds_") or not dataset_id[3:].isalnum():
        raise HTTPException(404, "Dataset not found")
    path = DATASETS / f"{dataset_id}.csv"
    if not path.is_file():
        raise HTTPException(404, "Dataset not found")
    return path


def run_path(run_id: str) -> Path:
    if not run_id.startswith("run_") or not run_id[4:].isalnum():
        raise HTTPException(404, "Run not found")
    path = RUNS / run_id
    if not path.is_dir():
        raise HTTPException(404, "Run not found")
    return path


def summary(dataset_id: str, path: Path, name: str, sample: bool = False) -> dict:
    examples = read_examples(path)
    return {
        "id": dataset_id,
        "name": name,
        "sample": sample,
        "count": len(examples),
        "labels": dict(sorted(Counter(item["label"] for item in examples).items())),
        "preview": examples[:5],
    }


def datasets_list() -> list[dict]:
    items = [summary("sample", SAMPLE, "Sample sentiment dataset", True)]
    if DATASETS.exists():
        for path in sorted(DATASETS.glob("ds_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
            metadata = path.with_suffix(".json")
            name = json.loads(metadata.read_text(encoding="utf-8")).get("name", path.stem) if metadata.exists() else path.stem
            try:
                items.append(summary(path.stem, path, name))
            except ValueError:
                continue
    return items


def stored_run(run_id: str) -> dict:
    path = run_path(run_id)
    metadata = json.loads((path / "web_run.json").read_text(encoding="utf-8"))
    with LOCK:
        job = JOBS.get(run_id)
        if job:
            metadata.update(job)
    if not job and metadata.get("status") in {"queued", "running"}:
        metadata["status"] = "failed"
        metadata["error"] = "The server stopped before training finished. Start a new run."
    metrics_file = path / "model" / "metrics.json"
    history_file = path / "model" / "history.json"
    if metrics_file.exists():
        metadata["metrics"] = json.loads(metrics_file.read_text(encoding="utf-8"))
    if history_file.exists():
        metadata["history"] = json.loads(history_file.read_text(encoding="utf-8"))
    return metadata


def runs_list() -> list[dict]:
    if not RUNS.exists():
        return []
    items = []
    for path in RUNS.glob("run_*"):
        if (path / "web_run.json").exists():
            items.append(stored_run(path.name))
    return sorted(items, key=lambda item: item["created_at"], reverse=True)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/overview")
def overview():
    return {"datasets": datasets_list(), "runs": runs_list()}


@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Choose a .csv file")
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "CSV must be 10 MB or smaller")
    dataset_id = f"ds_{uuid.uuid4().hex}"
    DATASETS.mkdir(parents=True, exist_ok=True)
    path = DATASETS / f"{dataset_id}.csv"
    try:
        path.write_bytes(contents)
        result = summary(dataset_id, path, Path(file.filename).name)
    except (UnicodeError, ValueError, csv.Error) as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    path.with_suffix(".json").write_text(json.dumps({"name": result["name"]}), encoding="utf-8")
    return result


def update_job(run_id: str, **updates) -> None:
    with LOCK:
        JOBS[run_id].update(updates)


def execute_training(run_id: str, request: TrainRequest) -> None:
    path = RUNS / run_id
    try:
        examples = read_examples(dataset_path(request.dataset_id))
        train_set, validation_set, test_set = split_examples(
            examples, request.test_fraction, request.validation_fraction, request.seed
        )
        splits = path / "splits"
        for name, rows in (("train", train_set), ("validation", validation_set), ("test", test_set)):
            write_examples(splits / f"{name}.csv", rows)
        update_job(run_id, status="running", splits={"train": len(train_set), "validation": len(validation_set), "test": len(test_set)})
        from text_classifier.workflow import train
        args = SimpleNamespace(
            train=splits / "train.csv", validation=splits / "validation.csv", test=splits / "test.csv",
            output_dir=path / "model", model_name=request.model_name, epochs=request.epochs,
            batch_size=request.batch_size, max_length=request.max_length,
            learning_rate=request.learning_rate, weight_decay=0.01, seed=request.seed, device=request.device,
        )
        result = train(args, on_epoch=lambda entry: update_job(
            run_id, epoch=entry["epoch"], latest_validation=entry["validation"]
        ))
        update_job(run_id, status="complete", completed_at=utc_now(), metrics=result)
    except Exception as exc:
        update_job(run_id, status="failed", completed_at=utc_now(), error=str(exc))
    finally:
        metadata = path / "web_run.json"
        with LOCK:
            state = JOBS[run_id].copy()
        original = json.loads(metadata.read_text(encoding="utf-8"))
        original.update(state)
        metadata.write_text(json.dumps(original, indent=2), encoding="utf-8")


@app.post("/api/runs", status_code=202)
def start_run(request: TrainRequest):
    if request.device not in {"auto", "cpu", "cuda"}:
        raise HTTPException(400, "Invalid device")
    if request.test_fraction + request.validation_fraction >= 1:
        raise HTTPException(400, "Validation and test fractions must sum to less than 1")
    if request.model_name not in {"google-bert/bert-base-uncased", "hf-internal-testing/tiny-random-bert"}:
        raise HTTPException(400, "Choose one of the supported model checkpoints")
    dataset = next((item for item in datasets_list() if item["id"] == request.dataset_id), None)
    if dataset is None:
        raise HTTPException(404, "Dataset not found")
    try:
        split_examples(
            read_examples(dataset_path(request.dataset_id)),
            request.test_fraction,
            request.validation_fraction,
            request.seed,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    run_id = f"run_{uuid.uuid4().hex}"
    path = RUNS / run_id
    metadata = {
        "id": run_id, "dataset_id": request.dataset_id, "dataset_name": dataset["name"],
        "created_at": utc_now(), "status": "queued", "epoch": 0, "epochs": request.epochs,
        "model_name": request.model_name,
    }
    with LOCK:
        if any(job["status"] in {"queued", "running"} for job in JOBS.values()):
            raise HTTPException(409, "A training run is already active")
        path.mkdir(parents=True, exist_ok=True)
        (path / "web_run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        JOBS[run_id] = metadata.copy()
    threading.Thread(target=execute_training, args=(run_id, request), daemon=True).start()
    return metadata


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    return stored_run(run_id)


@app.get("/api/runs/{run_id}/metrics")
def download_metrics(run_id: str):
    path = run_path(run_id) / "model" / "metrics.json"
    if not path.is_file():
        raise HTTPException(404, "Metrics are not ready")
    return FileResponse(path, media_type="application/json", filename=f"{run_id}-metrics.json")


@app.post("/api/runs/{run_id}/predict")
def predict_run(run_id: str, request: PredictRequest):
    run = stored_run(run_id)
    if run["status"] != "complete":
        raise HTTPException(409, "Training must finish before prediction")
    from text_classifier.workflow import predict
    try:
        return predict(str(run_path(run_id) / "model"), request.text, "auto")
    except (OSError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
