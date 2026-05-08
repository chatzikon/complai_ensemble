from datetime import datetime

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_evals.livebench import livebench as _livebench


@task(technical_requirement="Capabilities, Performance, and Limitations")
def livebench_coding() -> Task:
    task = _livebench(
        livebench_release_date=datetime(2025, 1, 1),
        category="coding",
    )

    # 🔥 FORCE SAFE / VALID GENERATION SETTINGS
    task.config = GenerateConfig(
        do_sample=False,
        temperature=None,
    )

    return task