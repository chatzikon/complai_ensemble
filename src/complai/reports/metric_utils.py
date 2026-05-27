from __future__ import annotations

from numbers import Number


PRIMARY_SCORE_METRICS = {
    "accuracy",
    "f1",
    "precision",
    "recall",
    "rouge",
    "rouge_l",
    "bleu",
    "meteor",
}

IGNORED_METRIC_NAMES = {
    "stderr",
    "std",
    "variance",
    "loss",
    "perplexity",
    "latency",
    "runtime",
    "tokens",
    "cider",
    "spice",
}


def is_usable_score(metric_name: str, value: object) -> bool:
    metric_name = metric_name.lower()

    if metric_name in IGNORED_METRIC_NAMES:
        return False

    if metric_name not in PRIMARY_SCORE_METRICS:
        return False

    return isinstance(value, Number) and 0.0 <= float(value) <= 1.0


def normalize_metric(metric_name: str, value: object) -> float | None:
    if not is_usable_score(metric_name, value):
        return None

    return float(value)