from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


RESULTS_DIR = Path("reports/model_runs")


def safe_model_name(model_name: str) -> str:
    return (
        model_name.replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def save_evaluation_result(
    model_name: str,
    evaluation_name: str,
    result: dict[str, Any],
) -> Path:
    model_dir = RESULTS_DIR / safe_model_name(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    output_path = model_dir / f"{evaluation_name}.json"

    payload = {
        "model_name": model_name,
        "evaluation_name": evaluation_name,
        "timestamp": datetime.now().isoformat(),
        "result": result,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return output_path