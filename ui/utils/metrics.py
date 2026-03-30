from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_logs_json(log_dir: str | Path) -> dict[str, Any] | None:
    log_dir = Path(log_dir)
    log_file = log_dir / "logs.json"

    if not log_file.exists():
        return None

    with log_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics_from_log_dir(log_dir: str | Path) -> dict[str, float]:
    payload = load_logs_json(log_dir)
    if payload is None or not payload:
        return {}

    first_run = next(iter(payload.values()), None)
    if not isinstance(first_run, dict):
        return {}

    results = first_run.get("results", {})
    scores = results.get("scores", [])

    extracted: dict[str, float] = {}

    for score_entry in scores:
        metrics = score_entry.get("metrics", {})
        if not isinstance(metrics, dict):
            continue

        for metric_key, metric_info in metrics.items():
            if not isinstance(metric_info, dict):
                continue

            value = metric_info.get("value")
            if isinstance(value, (int, float)):
                extracted[metric_key] = float(value)

    return extracted


def prettify_metric_name(name: str) -> str:
    return name.replace("_", " ").upper()