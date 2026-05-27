import streamlit as st

from complai.reports.storage import delete_model_results


def _format_score(score):
    if score is None:
        return "N/A"
    return f"{score:.2f}"


def _safe_progress_value(score):
    if score is None:
        return 0.0
    return max(0.0, min(float(score), 1.0))


def _safe_key(value: str) -> str:
    return (
        value.replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def render_model_report_panel(
    report: dict | None,
    model_name: str | None = None,
    key_prefix: str = "report",
    show_reset: bool = True,
):
    st.markdown("### Model Report")

    if model_name:
        st.caption(f"Model: `{model_name}`")

    if not report:
        st.info("No report available yet for this model.")
        return

    overall_score = report.get("overall", {}).get("score")

    st.metric("Overall Score", _format_score(overall_score))
    st.progress(_safe_progress_value(overall_score))

    by_principle = report.get("by_principle", {})
    by_requirement = report.get("by_requirement", {})
    by_benchmark = report.get("by_benchmark", {})

    st.divider()
    st.markdown("#### Detailed Breakdown")

    if not by_principle:
        st.info("No principle scores available yet.")
    else:
        for principle_id, principle in by_principle.items():
            principle_name = principle.get("principle_name", principle_id)
            principle_score = principle.get("score")

            with st.expander(
                f"{principle_name} — {_format_score(principle_score)}",
                expanded=False,
            ):
                st.progress(_safe_progress_value(principle_score))

                matching_requirements = [
                    requirement
                    for requirement in by_requirement.values()
                    if requirement.get("principle_id") == principle_id
                ]

                if not matching_requirements:
                    st.caption("No technical requirements available.")
                    continue

                for requirement in matching_requirements:
                    requirement_id = requirement.get("requirement_id")
                    requirement_name = requirement.get(
                        "requirement_name",
                        requirement_id,
                    )
                    requirement_score = requirement.get("score")

                    st.markdown(
                        f"**{requirement_name}** — "
                        f"{_format_score(requirement_score)}"
                    )
                    st.progress(_safe_progress_value(requirement_score))

                    extra = []

                    if "num_benchmarks" in requirement:
                        extra.append(
                            f"{requirement['num_benchmarks']} benchmark(s)"
                        )

                    if "num_questions" in requirement:
                        extra.append(
                            f"{requirement['num_questions']} question(s)"
                        )

                    if "not_applicable_questions" in requirement:
                        extra.append(
                            f"{requirement['not_applicable_questions']} N/A"
                        )

                    if extra:
                        st.caption(" | ".join(extra))

                    matching_benchmarks = [
                        benchmark
                        for benchmark in by_benchmark.values()
                        if benchmark.get("requirement_id") == requirement_id
                    ]

                    if matching_benchmarks:
                        for benchmark in matching_benchmarks:
                            benchmark_name = benchmark.get(
                                "benchmark_name",
                                "Unknown benchmark",
                            )
                            benchmark_score = benchmark.get("score")

                            st.markdown(
                                f"- `{benchmark_name}` — "
                                f"{_format_score(benchmark_score)}"
                            )

    if show_reset and model_name:
        st.divider()

        safe_key_model = _safe_key(model_name)

        confirm_reset = st.checkbox(
            "Confirm reset",
            key=f"{key_prefix}_confirm_reset_{safe_key_model}",
        )

        if st.button(
            "Reset saved results for this model",
            disabled=not confirm_reset,
            key=f"{key_prefix}_reset_results_{safe_key_model}",
        ):
            deleted = delete_model_results(model_name)

            if deleted:
                st.success("Saved results were deleted. Refreshing...")
                st.rerun()
            else:
                st.info("No saved results found for this model.")