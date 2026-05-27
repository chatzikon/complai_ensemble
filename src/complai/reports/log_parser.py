from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from complai.reports.metric_utils import normalize_metric


def load_logs_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_metric_results_from_logs(
    logs: dict[str, Any],
    task_to_category: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    metric_results: list[dict[str, Any]] = []

    task_to_category = task_to_category or {}

    for _, run in logs.items():
        if run.get("status") != "success":
            continue

        eval_info = run.get("eval", {})
        results = run.get("results", {})

        model_name = eval_info.get("model")
        task_name = eval_info.get("task")
        task_attribs = eval_info.get("task_attribs", {})

        requirement_name = task_attribs.get("technical_requirement")
        principle_name = task_to_category.get(task_name, "Unknown")

        if not model_name or not task_name or not requirement_name:
            continue

        for score_block in results.get("scores", []):
            for metric_name, metric_data in score_block.get("metrics", {}).items():
                raw_value = metric_data.get("value")
                normalized_score = normalize_metric(metric_name, raw_value)

                if normalized_score is None:
                    print(f"Skipping metric '{metric_name}' with value {raw_value}")
                    continue

                metric_results.append(
                    {
                        "model_name": model_name,
                        "task": task_name,
                        "technical_requirement": requirement_name,
                        "metric_name": metric_name,
                        "raw_value": raw_value,
                        "normalized_score": normalized_score,
                    }
                )

    return metric_results