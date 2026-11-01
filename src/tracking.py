"""Experiment tracking: MLflow when available, plus a plain JSONL run log.

Every training run is appended to results/runs.jsonl so the numbers in the
README are reproducible even without an MLflow server.
"""
import json
import time

from . import config

try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:  # pragma: no cover - environment without mlflow
    _HAS_MLFLOW = False

_RUNS_LOG = config.RESULTS_DIR / "runs.jsonl"


def setup() -> None:
    if _HAS_MLFLOW:
        mlflow.set_tracking_uri("sqlite:///" + str(config.ROOT / "mlflow.db"))
        mlflow.set_experiment("rec-engine")


def log_run(model: str, params: dict, metrics: dict, extra: dict | None = None) -> None:
    """Record one training/eval run in MLflow and results/runs.jsonl."""
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model,
        "params": params,
        "metrics": metrics,
        "extra": extra or {},
    }
    _RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _RUNS_LOG.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    if _HAS_MLFLOW:
        with mlflow.start_run(run_name=model):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metrics.items() if isinstance(v, (int, float))})
