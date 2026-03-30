from __future__ import annotations

from inspect_ai.scorer import Score, Target, metric, scorer, SampleScore
from inspect_ai.solver import TaskState

from sklearn.metrics import accuracy_score, f1_score, precision_score


def _extract(sample_scores: list[SampleScore]):
    y_true = []
    y_pred = []

    for sample_score in sample_scores:
        score = sample_score.score
        meta = score.metadata or {}
        y_true.append(str(meta["reference"]).strip().lower())
        y_pred.append(str(meta["prediction"]).strip().lower())

    return y_true, y_pred


@metric
def classification_accuracy():
    def compute(sample_scores: list[SampleScore]) -> float:
        y_true, y_pred = _extract(sample_scores)
        return float(accuracy_score(y_true, y_pred))
    return compute


@metric
def classification_f1_macro():
    def compute(sample_scores: list[SampleScore]) -> float:
        y_true, y_pred = _extract(sample_scores)
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return compute


@metric
def classification_precision_macro():
    def compute(sample_scores: list[SampleScore]) -> float:
        y_true, y_pred = _extract(sample_scores)
        return float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    return compute


@scorer(
    metrics=[
        classification_accuracy(),
        classification_f1_macro(),
        classification_precision_macro(),
    ]
)
def classification_scorer():
    async def score(state: TaskState, target: Target):
        prediction = (state.output.completion or "").strip().lower()
        reference = target.text if hasattr(target, "text") else target

        return Score(
            value=0.0,
            answer=prediction,
            metadata={
                "prediction": prediction,
                "reference": str(reference).strip().lower(),
            },
        )

    return score