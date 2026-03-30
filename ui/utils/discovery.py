from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def discover_tasks() -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    """
    Returns:
        - task_list: flat list of task names
        - task_to_category: mapping task -> category
        - category_to_tasks: mapping category -> [tasks]
    """

    try:
        result = subprocess.run(
            ["complai", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return [], {}, {}

    task_list: List[str] = []
    task_to_category: Dict[str, str] = {}
    category_to_tasks: Dict[str, List[str]] = {}

    current_category = None

    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        # Category line (no indentation)
        if not raw_line.startswith("  "):
            current_category = line
            if current_category not in category_to_tasks:
                category_to_tasks[current_category] = []
            continue

        # Task line (indented)
        if current_category is None:
            continue

        tasks = [t.strip() for t in line.split(",") if t.strip()]

        for task in tasks:
            task_list.append(task)
            task_to_category[task] = current_category
            category_to_tasks[current_category].append(task)

    return sorted(set(task_list)), task_to_category, category_to_tasks


def discover_local_models() -> list[str]:
    models = []

    models_dir = Path("src/complai/models")

    if (models_dir / "malware_gcn_provider.py").exists():
        models.append("malware/baseline_gcn")

    if (models_dir / "multimodal.py").exists():
        models.append("multimodal/my_model")

    return sorted(set(models))


def discover_all_models() -> list[str]:
    curated = [
        "openai/gpt-4o-mini",
        "openai/gpt-5-nano",
        "anthropic/claude-sonnet-4-0",
    ]

    return sorted(set(discover_local_models() + curated))