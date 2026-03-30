from __future__ import annotations

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from inspect_ai.scorer import Score, Target, scorer, metric, SampleScore
from inspect_ai.solver import TaskState
import subprocess
from contextlib import contextmanager


@contextmanager
def suppress_subprocess_output():
    original_popen = subprocess.Popen

    def quiet_popen(*args, **kwargs):
        kwargs.setdefault("stdout", subprocess.DEVNULL)
        kwargs.setdefault("stderr", subprocess.DEVNULL)
        return original_popen(*args, **kwargs)

    subprocess.Popen = quiet_popen
    try:
        yield
    finally:
        subprocess.Popen = original_popen



def _normalize_prediction(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_reference(value) -> list[str]:
    if value is None:
        return [""]

    if isinstance(value, list):
        normalized = [str(v).strip() for v in value if v is not None]
        return normalized if normalized else [""]

    return [str(value).strip()]




def _build_coco_dict(sample_scores):
    preds = {}
    refs = {}

    for i, sample_score in enumerate(sample_scores):
        score = sample_score.score
        meta = score.metadata or {}

        prediction = _normalize_prediction(meta.get("prediction", ""))
        reference = _normalize_reference(meta.get("reference", ""))

        preds[i] = [prediction]
        refs[i] = reference

    return preds, refs





@metric
def bleu_1():
    def compute(sample_scores: list[SampleScore]) -> float:

        preds, refs = _build_coco_dict(sample_scores)

        bleu_scores, _ = Bleu(4).compute_score(refs, preds)
        return float(bleu_scores[0])
    return compute


@metric
def bleu_2():
    def compute(sample_scores: list[SampleScore]) -> float:
        preds, refs = _build_coco_dict(sample_scores)
        bleu_scores, _ = Bleu(4).compute_score(refs, preds)
        return float(bleu_scores[1])
    return compute


@metric
def bleu_3():
    def compute(sample_scores: list[SampleScore]) -> float:
        preds, refs = _build_coco_dict(sample_scores)
        bleu_scores, _ = Bleu(4).compute_score(refs, preds)
        return float(bleu_scores[2])
    return compute


@metric
def bleu_4():
    def compute(sample_scores: list[SampleScore]) -> float:
        preds, refs = _build_coco_dict(sample_scores)
        bleu_scores, _ = Bleu(4).compute_score(refs, preds)
        return float(bleu_scores[3])
    return compute


@metric
def rouge_l():
    def compute(sample_scores: list[SampleScore]) -> float:
        preds, refs = _build_coco_dict(sample_scores)
        rouge_score, _ = Rouge().compute_score(refs, preds)
        return float(rouge_score)
    return compute


@metric
def cider():
    def compute(sample_scores: list[SampleScore]) -> float:
        preds, refs = _build_coco_dict(sample_scores)
        cider_score, _ = Cider().compute_score(refs, preds)
        return float(cider_score)
    return compute


@metric
def spice():
    def compute(sample_scores: list[SampleScore]) -> float:
        preds, refs = _build_coco_dict(sample_scores)
        with suppress_subprocess_output():
            spice_score, _ = Spice().compute_score(refs, preds)
        return float(spice_score)
    return compute


@scorer(metrics=[bleu_1(), bleu_2(), bleu_3(), bleu_4(), rouge_l(), cider(), spice()])
def caption_sample():
    async def score(state: TaskState, target: Target):
        prediction = (state.output.completion or "").strip()
        reference = target.text if hasattr(target, "text") else target




        return Score(
            value=0.0,  # numeric placeholder avoids float-conversion warnings
            answer=prediction,
            metadata={
                "prediction": prediction,
                "reference": reference,
            },
        )

    return score