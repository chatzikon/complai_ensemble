import subprocess
import re
from typing import List, Dict, Tuple
from pathlib import Path

CATEGORIES = [
    "Robustness and Predictability",
    "Cyberattack Resilience",
    "Training Data Suitability",
    "No Copyright Infringement",
    "User Privacy Protection"
    "Capabilities, Performance, and Limitations",
    "Interpretability",
    "Disclosure of AI",
    "Traceability",
    "Fairness — Absence of Discrimination",
    "Representation — Absence of Bias",
    "Enviromental Impact",
    "Harmful Content and Toxicity",
]


def clean_ansi_codes(text: str) -> str:
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def unwrap_terminal_lines(text: str) -> str:
    """
    Fix HARD WRAPPING caused by terminal width.
    Key rule:
    - if line does NOT start with a category AND does NOT contain category
      and previous line doesn't end with comma → merge
    """

    lines = text.splitlines()
    rebuilt = []

    for line in lines:
        line = line.rstrip()

        if not line:
            continue

        # category line stays
        if line in CATEGORIES:
            rebuilt.append(line)
            continue

        # continuation of previous line (no leading category)
        if rebuilt:
            prev = rebuilt[-1]

            # if previous line looks incomplete (no category + not ending cleanly)
            if (
                prev not in CATEGORIES
                and not prev.endswith(",")
                and not line.strip() in CATEGORIES
            ):
                rebuilt[-1] = prev + line.strip()
                continue

        rebuilt.append(line)

    return "\n".join(rebuilt)


def discover_tasks():

    try:
        result = subprocess.run(
            ["complai", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as e:
        print(f"[discover_tasks] error: {e}")
        return [], {}, {}

    text = clean_ansi_codes(result.stdout)
    text = unwrap_terminal_lines(text)

    task_list = []
    task_to_category = {}
    category_to_tasks = {c: [] for c in CATEGORIES}

    current_category = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # category detection
        if line in CATEGORIES:
            current_category = line
            continue

        if current_category is None:
            continue

        # task parsing
        for task in [t.strip() for t in line.split(",") if t.strip()]:
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