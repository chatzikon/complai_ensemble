from pathlib import Path
from typing import Any

import streamlit as st

from complai.scorers.altai_qualitative_metrics import altai_qualitative_scorer
from complai.solvers.altai_qualitative_solver import AltaiQualitativeSolver
from complai.tasks.altai_qualitative.altai_qualitative import (
    build_altai_samples,
    load_altai_questions,
)


SCORE_LABEL_TO_VALUE = {
    "N/A": None,
    "Not Implemented": 0,
    "Partially Implemented": 1,
    "Fully Implemented": 2,
}

SCORE_VALUE_TO_LABEL = {
    value: label for label, value in SCORE_LABEL_TO_VALUE.items()
}


def _render_bullets(title: str, items: list[str]) -> None:
    if not items:
        return

    with st.expander(title, expanded=False):
        for item in items:
            st.markdown(f"- {item}")


def _render_scoring_guidelines(guidelines: dict[Any, str]) -> None:
    if not guidelines:
        return

    with st.expander("Scoring guidelines", expanded=False):
        for score in sorted(guidelines, key=lambda x: int(x)):
            label = SCORE_VALUE_TO_LABEL.get(int(score), str(score))
            st.markdown(f"**{score} — {label}:** {guidelines[score]}")


def render_altai_qualitative_page(
    questionnaire_path: str | Path = "config/altai/qualitative_questions_v1.yaml",
) -> dict[str, Any] | None:
    st.header("Qualitative Evaluation")
    st.caption("N/A = Not applicable to this AI system.")
    st.caption(
        "Questionnaire-based governance assessment for Human Agency & Oversight "
        "and Accountability."
    )

    questionnaire = load_altai_questions(questionnaire_path)
    samples = build_altai_samples(questionnaire)

    responses: dict[str, dict[str, Any]] = {}

    for sample in samples:
        st.divider()

        st.subheader(
            f"{sample['principle_name']} — {sample['requirement_name']}"
        )

        st.markdown(f"**{sample['question_id']}**")
        st.write(sample["question"])

        _render_bullets(
            "Supporting indicators",
            sample.get("supporting_indicators", []),
        )

        # _render_bullets(
        #     "Expected evidence",
        #     sample.get("expected_evidence", []),
        # )

        _render_scoring_guidelines(
            sample.get("scoring_guidelines", {})
        )

        selected_label = st.radio(
            label=f"Score for {sample['question_id']}",
            options=list(SCORE_LABEL_TO_VALUE.keys()),
            key=f"score_{sample['question_id']}",
            horizontal=True,
        )

        justification = st.text_area(
            label=f"Justification for {sample['question_id']} optional",
            key=f"justification_{sample['question_id']}",
        )

        responses[sample["question_id"]] = {
            "score": SCORE_LABEL_TO_VALUE[selected_label],
            "justification": justification.strip(),
        }

    if st.button("Evaluate qualitative score"):
        solver = AltaiQualitativeSolver(responses)
        predictions = [solver(sample) for sample in samples]

        results = altai_qualitative_scorer(samples, predictions)

        st.subheader("Results")
        st.json(results)

        return results

    return None