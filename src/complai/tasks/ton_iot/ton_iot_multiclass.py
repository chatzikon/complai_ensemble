from inspect_ai import task, Task
from inspect_ai.dataset import Sample

from complai.datasets.ton_iot_network_dataset import load_ton_iot_network
from complai.scorers.classification_metrics import classification_scorer


@task(technical_requirement="Robustness and Predictability")
def ton_iot_multiclass():
    rows = load_ton_iot_network(split="test[:1000]", label_mode="multiclass")

    samples = [
        Sample(
            input=row["input"],
            target=row["target"],
            metadata=row["metadata"],
        )
        for row in rows
    ]

    return Task(
        dataset=samples,
        scorer=classification_scorer(),
    )