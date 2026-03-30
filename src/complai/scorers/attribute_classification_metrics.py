from __future__ import annotations

import re
import numpy as np

from inspect_ai.scorer import Score, Target, metric, scorer, SampleScore
from inspect_ai.solver import TaskState
from sklearn.metrics import confusion_matrix


ATTRIBUTE_VOCAB = [
    "young",
    "male",
    "female",
    "smiling",
    "eyeglasses",
    "black_hair",
    "blond_hair",
    "bald",
    "mustache",
    "wearing_lipstick",
]


def normalize_attr(attr: str) -> str:
    return attr.strip().lower().replace(" ", "_")


def parse_prediction_to_set(text: str) -> set[str]:
    text = str(text).lower()
    found = set()

    # exact comma/newline/semicolon split
    parts = re.split(r"[,\n;]+", text)
    for part in parts:
        cleaned = normalize_attr(part)
        if cleaned in ATTRIBUTE_VOCAB:
            found.add(cleaned)

    # substring fallback
    for attr in ATTRIBUTE_VOCAB:
        if attr in text:
            found.add(attr)

    aliases = {
        "black hair": "black_hair",
        "blond hair": "blond_hair",
        "wearing lipstick": "wearing_lipstick",
    }
    for phrase, canonical in aliases.items():
        if phrase in text:
            found.add(canonical)

    return found


def normalize_reference(reference) -> set[str]:
    if reference is None:
        return set()

    if isinstance(reference, list):
        return {normalize_attr(x) for x in reference if x is not None}

    return {normalize_attr(str(reference))}


def to_binary_vector(attr_set: set[str]) -> list[int]:
    return [1 if attr in attr_set else 0 for attr in ATTRIBUTE_VOCAB]


def _extract_flat_binary_labels(sample_scores: list[SampleScore]):
    true_labels = []
    predicted_labels = []

    for sample_score in sample_scores:
        score = sample_score.score
        meta = score.metadata or {}

        ref_set = normalize_reference(meta["reference"])
        pred_set = parse_prediction_to_set(meta["prediction"])

        ref_vec = to_binary_vector(ref_set)
        pred_vec = to_binary_vector(pred_set)

        true_labels.extend(ref_vec)
        predicted_labels.extend(pred_vec)

    return true_labels, predicted_labels


def _binary_confusion_stats(sample_scores: list[SampleScore]):
    true_labels, predicted_labels = _extract_flat_binary_labels(sample_scores)

    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])

    # Ensure 2x2
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # defensive fallback
        tn = fp = fn = tp = 0
        if len(cm) > 0:
            if cm.shape[0] > 0 and cm.shape[1] > 0:
                tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
            if cm.shape[0] > 0 and cm.shape[1] > 1:
                fp = cm[0, 1]
            if cm.shape[0] > 1 and cm.shape[1] > 0:
                fn = cm[1, 0]
            if cm.shape[0] > 1 and cm.shape[1] > 1:
                tp = cm[1, 1]

    return tn, fp, fn, tp


@metric
def multilabel_accuracy():
    def compute(sample_scores: list[SampleScore]) -> float:
        tn, fp, fn, tp = _binary_confusion_stats(sample_scores)
        denom = tp + tn + fp + fn
        return float((tp + tn) / denom) if denom > 0 else 0.0
    return compute


@metric
def multilabel_precision():
    def compute(sample_scores: list[SampleScore]) -> float:
        tn, fp, fn, tp = _binary_confusion_stats(sample_scores)
        denom = tp + fp
        return float(tp / denom) if denom > 0 else 0.0
    return compute


@metric
def multilabel_recall():
    def compute(sample_scores: list[SampleScore]) -> float:
        tn, fp, fn, tp = _binary_confusion_stats(sample_scores)
        denom = tp + fn
        return float(tp / denom) if denom > 0 else 0.0
    return compute


@metric
def multilabel_f1():
    def compute(sample_scores: list[SampleScore]) -> float:
        tn, fp, fn, tp = _binary_confusion_stats(sample_scores)
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        denom = precision + recall
        return float(2 * (precision * recall) / denom) if denom > 0 else 0.0
    return compute


@scorer(
    metrics=[
        multilabel_accuracy(),
        multilabel_precision(),
        multilabel_recall(),
        multilabel_f1(),
    ]
)
def attribute_classification_scorer():
    async def score(state: TaskState, target: Target):
        prediction = (state.output.completion or "").strip()
        reference = state.metadata["gold_attributes"]

        return Score(
            value=0.0,
            answer=prediction,
            metadata={
                "prediction": prediction,
                "reference": reference,
            },
        )

    return score