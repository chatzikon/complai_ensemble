from pathlib import Path

from complai._cli.utils import get_complai_tasks

TECHNICAL_REQUIREMENT_TO_PRINCIPLE = {
    "No Technical Requirements": "Human Agency and Oversight",

    "Robustness and Predictability": "Technical Robustness and Safety",
    "Cyberattack Resilience": "Technical Robustness and Safety",

    "Training Data Suitability": "Privacy and Data Governance",
    "No Copyright Infringement": "Privacy and Data Governance",
    "User Privacy Protection": "Privacy and Data Governance",

    "Capabilities, Performance, and Limitations": "Transparency",
    "Interpretability": "Transparency",
    "Disclosure of AI Presence": "Transparency",
    "Traceability": "Transparency",

    "Fairness — Absence of Discrimination":
        "Diversity, Non-discrimination and Fairness",
    "Representation — Absence of Bias":
        "Diversity, Non-discrimination and Fairness",

    "Societal Alignment": "Societal and Environmental Well-being",
    "Environmental Impact": "Societal and Environmental Well-being",
    "Harmful Content and Toxicity": "Societal and Environmental Well-being",
}


def discover_tasks() -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    """Discover the same tasks as the CLI without parsing terminal output.

    Inspect reads @task declarations under src/complai/tasks. Categories come
    directly from their technical_requirement attributes, not a UI allowlist.
    Let discovery errors reach the UI so they are visible to the user.
    """
    task_to_category = {
        task.name: task.attribs.get("technical_requirement") or "Uncategorized"
        for task in get_complai_tasks()
    }
    task_list = sorted(task_to_category)
    category_to_tasks: dict[str, list[str]] = {
        category: [] for category in sorted(set(task_to_category.values()))
    }
    for task_name in task_list:
        category_to_tasks[task_to_category[task_name]].append(task_name)

    return task_list, task_to_category, category_to_tasks


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