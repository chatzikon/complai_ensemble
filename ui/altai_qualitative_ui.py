from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from config import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    LOCAL_PROVIDERS,
    PROVIDER_DEFAULT_MODELS,
)


from complai.reports.model_report import build_model_report_from_saved_results
from complai.reports.storage import (
    save_evaluation_result,
    load_model_results,
)


from complai.scorers.altai_qualitative_metrics import altai_qualitative_scorer
from complai.solvers.altai_qualitative_solver import AltaiQualitativeSolver
from complai.tasks.altai_qualitative.altai_qualitative import (
    build_altai_samples,
    load_altai_questions,
)


from report_panel import render_model_report_panel

from ui.report_panel import render_model_report_panel

def safe_widget_key(value: str) -> str:
    return (
        value.replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
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

def get_model_state_key(provider: str) -> str:
    return f"selected_model_name__{provider}"

def build_model_string(provider: str, model_name: str) -> str:
    model_name = model_name.strip()

    if not model_name:
        return provider

    return f"{provider}/{model_name}"


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

    if "last_altai_success" in st.session_state:
        st.success(st.session_state.pop("last_altai_success"))



    col_provider, col_model = st.columns(2)

    with col_provider:
        provider = st.selectbox(
            "Provider",
            LOCAL_PROVIDERS,
            index=LOCAL_PROVIDERS.index(
                st.session_state.get("selected_provider", DEFAULT_PROVIDER)
            )
            if st.session_state.get("selected_provider", DEFAULT_PROVIDER) in LOCAL_PROVIDERS
            else 0,
            key="altai_provider",
        )

    st.session_state["selected_provider"] = provider

    default_model_for_provider = PROVIDER_DEFAULT_MODELS.get(provider, DEFAULT_MODEL)
    model_state_key = get_model_state_key(provider)

    if model_state_key not in st.session_state:
        st.session_state[model_state_key] = default_model_for_provider

    # current_model_value = st.session_state.get(
    #     "selected_model_name",
    #     default_model_for_provider,
    # )

    with col_model:
        model_name = st.text_input(
            "Model name",
            value=st.session_state[model_state_key],
            help="Use the same model identifier as in quantitative evaluation.",
            key=f"altai_model_name_{provider}",
        )

    st.session_state[model_state_key] = model_name

    model_spec = build_model_string(provider, model_name)

    main_col, report_col = st.columns([1.6, 1])



    with report_col:
        report_placeholder = st.empty()

    saved_results = load_model_results(model_spec)
    report_to_render = build_model_report_from_saved_results(saved_results)


    questionnaire = load_altai_questions(questionnaire_path)
    samples = build_altai_samples(questionnaire)



    responses: dict[str, dict[str, Any]] = {}

    with main_col:
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

            saved_path = save_evaluation_result(
                model_name=model_spec,
                evaluation_name="altai_qualitative",
                result=results,
            )

            st.session_state["last_altai_success"] = (
                f"Saved ALTAI result to {saved_path}"
            )
            st.session_state["last_altai_results"] = results

            st.rerun()

        if "last_altai_success" in st.session_state:
            st.success(st.session_state.pop("last_altai_success"))

        if "last_altai_results" in st.session_state:
            st.subheader("ALTAI Results")
            st.json(st.session_state["last_altai_results"])


    with report_placeholder.container():
        render_model_report_panel(
            report_to_render,
            model_spec,
            key_prefix=f"altai_report_{safe_widget_key(model_spec)}",
            show_reset=True,
        )

