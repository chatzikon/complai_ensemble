from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from complai._cli.utils import get_log_dir
from complai.reports.log_parser import (
    extract_metric_results_from_logs,
    load_logs_json,
)
from complai.reports.model_report import build_model_report_from_saved_results
from complai.reports.storage import (
    save_evaluation_result,
    load_model_results,
)

from config import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TASK,
    LOCAL_PROVIDERS,
    PROVIDER_DEFAULT_MODELS,
)

from utils.devices import default_device_value, discover_device_options
from utils.discovery import (
    discover_tasks,
    TECHNICAL_REQUIREMENT_TO_PRINCIPLE,
)
from utils.metrics import extract_metrics_from_log_dir, prettify_metric_name
from utils.runner import (
    stream_command,
    kill_process_tree,
    kill_running_evaluation,
    clear_running_evaluation_pid,
)

from altai_qualitative_ui import render_altai_qualitative_page
from ui.report_panel import render_model_report_panel


def safe_widget_key(value: str) -> str:
    return (
        value.replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def find_latest_logs_json(log_dir: str | Path) -> Path | None:
    log_dir = Path(log_dir)

    candidates = list(log_dir.rglob("logs.json"))

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def get_model_state_key(provider: str) -> str:
    return f"selected_model_name__{provider}"


def round_dict_values(metrics, decimals=2):
    return {
        k: round(v, decimals) if isinstance(v, float) else v
        for k, v in metrics.items()
    }


def clean_ansi_codes(text):
    ansi_escape = re.compile(
        r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
    )
    return ansi_escape.sub("", text)


def build_model_string(provider: str, model_name: str) -> str:
    model_name = model_name.strip()

    if not model_name:
        return provider

    return f"{provider}/{model_name}"


def render_header():
    col_left, col_right = st.columns([6, 1])

    with col_left:
        st.title("Assessment of ENSEMBLE’s AI solutions")

    with col_right:
        logo_path = (
            Path(__file__).parent
            / "assets"
            / "ENSEMBLE-logo_VNtF8Gw.png"
        )

        if logo_path.exists():
            st.image(str(logo_path), width="stretch")


def render_quantitative_benchmark_page():

    # ---------------------------------------------------------
    # Discover benchmark tasks
    # ---------------------------------------------------------

    try:
        task_list, task_to_category, category_to_tasks = discover_tasks()

    except Exception as exc:
        st.error(f"Could not discover benchmark tasks: {exc}")
        return

    if not task_list:
        st.warning(
            "No benchmark tasks found. "
            "Check src/complai/tasks and run complai list."
        )
        return

    # ---------------------------------------------------------
    # Build hierarchy:
    #
    # Ethical Principle
    #       ↓
    # Technical Requirement
    #       ↓
    # Benchmark
    # ---------------------------------------------------------

    principle_to_requirements: dict[str, dict[str, list[str]]] = {}

    for requirement, tasks in category_to_tasks.items():

        principle = TECHNICAL_REQUIREMENT_TO_PRINCIPLE.get(
            requirement,
            "Unmapped Principle",
        )

        principle_to_requirements.setdefault(
            principle,
            {},
        )[requirement] = tasks

    # ---------------------------------------------------------
    # Determine default benchmark hierarchy
    # ---------------------------------------------------------

    default_requirement = task_to_category.get(DEFAULT_TASK)

    default_principle = TECHNICAL_REQUIREMENT_TO_PRINCIPLE.get(
        default_requirement,
        "Unmapped Principle",
    )

    # ---------------------------------------------------------
    # Provider
    # ---------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        current_provider = st.session_state.get(
            "selected_provider",
            DEFAULT_PROVIDER,
        )

        provider = st.selectbox(
            "Provider",
            LOCAL_PROVIDERS,
            index=(
                LOCAL_PROVIDERS.index(current_provider)
                if current_provider in LOCAL_PROVIDERS
                else 0
            ),
            key="quant_provider",
        )

    st.session_state["selected_provider"] = provider

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    default_model_for_provider = PROVIDER_DEFAULT_MODELS.get(
        provider,
        DEFAULT_MODEL,
    )

    model_state_key = get_model_state_key(provider)

    if model_state_key not in st.session_state:
        st.session_state[model_state_key] = default_model_for_provider

    with col2:

        model_name = st.text_input(
            "Model name",
            value=st.session_state[model_state_key],
            help="Examples: baseline_gcn, CMAlign, Qwen/Qwen3-8B",
            key=f"quant_model_name_{provider}",
        )

    st.session_state[model_state_key] = model_name

    # ---------------------------------------------------------
    # Nested benchmark selection
    # ---------------------------------------------------------

    st.markdown("### Benchmark Selection")

    principle_col, requirement_col, benchmark_col = st.columns(3)

    # ---------------------------------------------------------
    # 1. Ethical Principle
    # ---------------------------------------------------------

    principles = sorted(principle_to_requirements.keys())

    with principle_col:

        principle = st.selectbox(
            "EU AI Act Ethical Principle",
            principles,
            index=(
                principles.index(default_principle)
                if default_principle in principles
                else 0
            ),
        )

    # ---------------------------------------------------------
    # 2. Technical Requirement
    # ---------------------------------------------------------

    requirements = sorted(
        principle_to_requirements[principle].keys()
    )

    with requirement_col:

        requirement = st.selectbox(
            "Technical Requirement",
            requirements,
            index=(
                requirements.index(default_requirement)
                if default_requirement in requirements
                else 0
            ),
        )

    # ---------------------------------------------------------
    # 3. Benchmark
    # ---------------------------------------------------------

    benchmarks = sorted(
        principle_to_requirements[principle][requirement]
    )

    with benchmark_col:

        task = st.selectbox(
            "Benchmark",
            benchmarks,
            index=(
                benchmarks.index(DEFAULT_TASK)
                if DEFAULT_TASK in benchmarks
                else 0
            ),
        )

    # ---------------------------------------------------------
    # Selected hierarchy
    # ---------------------------------------------------------

    st.caption(
        f"Selected: **{principle} → {requirement} → {task}**"
    )

    # ---------------------------------------------------------
    # Runtime options
    # ---------------------------------------------------------

    device_options = discover_device_options()

    device_labels = [
        label
        for _, label in device_options
    ]

    device_label_to_value = {
        label: value
        for value, label in device_options
    }

    default_value = default_device_value(device_options)

    default_label = next(
        label
        for value, label in device_options
        if value == default_value
    )

    col3, col4, col5, col6 = st.columns(4)

    with col3:
        limit = st.text_input(
            "Sample limit (-l)",
            "",
        )

    with col4:
        debug = st.checkbox(
            "Debug (--debug)",
            value=True,
        )

    with col5:
        selected_device_label = st.selectbox(
            "Device",
            device_labels,
            index=device_labels.index(default_label),
            help="Detected automatically from the current machine",
        )

    with col6:
        max_connections = st.text_input(
            "Max connections",
            value="4",
            help=(
                "Controls model generation concurrency / effective "
                "batch size. Lower values reduce GPU memory usage."
            ),
        )

    selected_device = device_label_to_value[selected_device_label]

    # ---------------------------------------------------------
    # Build model specification
    # ---------------------------------------------------------

    model_spec = build_model_string(
        provider,
        model_name,
    )

    log_dir = get_log_dir(
        model_spec,
        "logs",
    )

    # ---------------------------------------------------------
    # Build CLI command
    # ---------------------------------------------------------

    cmd = [
        "complai",
        "eval",
        model_spec,
        "-t",
        task,
        "--log-dir",
        log_dir,
    ]

    if limit.strip():
        cmd += [
            "-l",
            limit.strip(),
        ]

    if debug:
        cmd.append("--debug")

    if selected_device.strip():
        cmd += [
            "-M",
            f"device={selected_device.strip()}",
        ]

    if max_connections.strip():
        cmd += [
            "--max-connections",
            max_connections.strip(),
        ]

    # ---------------------------------------------------------
    # CLI preview
    # ---------------------------------------------------------

    st.markdown("### CLI Preview")

    st.code(
        " ".join(cmd),
        language="bash",
    )

    st.caption(
        f"Run logs will be saved to: `{log_dir}`"
    )

    # ---------------------------------------------------------
    # Logs / Metrics / Report
    # ---------------------------------------------------------

    show_logs = st.checkbox(
        "Show logs",
        value=False,
    )

    metrics_col, report_col = st.columns(
        [1.6, 1]
    )

    with metrics_col:

        st.markdown("### 📊 Metrics")

        metrics_area = st.empty()

    with report_col:

        report_placeholder = st.empty()

    saved_results = load_model_results(
        model_spec
    )

    report_to_render = (
        build_model_report_from_saved_results(
            saved_results
        )
    )

    if show_logs:

        st.markdown("### 📟 Logs")

        log_area = st.empty()

    else:

        log_area = None

    st.divider()

    # ---------------------------------------------------------
    # Stop running evaluation
    # ---------------------------------------------------------

    if st.button(
        "🛑 Stop last running evaluation / free GPU memory"
    ):

        killed = kill_running_evaluation()

        if killed:
            st.success(
                "Stopped the last running evaluation process."
            )

        else:
            st.info(
                "No running evaluation process was found."
            )

        st.rerun()

    # ---------------------------------------------------------
    # Run evaluation
    # ---------------------------------------------------------

    if st.button("▶ Run Evaluation"):

        logs_list = []
        process = None

        try:

            iterator, process = stream_command(cmd)

            st.session_state[
                "running_eval_process"
            ] = process

            for line in iterator:

                clean_line = clean_ansi_codes(
                    line.strip("\n")
                )

                if not clean_line:
                    continue

                is_progress_bar = "%" in clean_line

                if is_progress_bar and logs_list:

                    if "%" in logs_list[-1]:

                        logs_list[-1] = clean_line

                    else:

                        logs_list.append(clean_line)

                else:

                    logs_list.append(clean_line)

                if (
                    show_logs
                    and log_area is not None
                ):

                    display_text = "\n".join(
                        logs_list[-100:]
                    )

                    log_area.code(
                        display_text,
                        language="text",
                    )

            process.wait()

            # -------------------------------------------------
            # Extract metrics
            # -------------------------------------------------

            metrics = extract_metrics_from_log_dir(
                log_dir
            )

            logs_json_path = find_latest_logs_json(
                log_dir
            )

            metric_results = []

            if logs_json_path is not None:

                logs_json = load_logs_json(
                    logs_json_path
                )

                metric_results = (
                    extract_metric_results_from_logs(
                        logs_json
                    )
                )

            # -------------------------------------------------
            # Debug information
            # -------------------------------------------------

            st.write(
                "DEBUG metric_results:",
                metric_results,
            )

            st.write(
                "DEBUG requirements:",
                [
                    r.get("technical_requirement")
                    or r.get("requirement_name")
                    or r.get("requirement_id")
                    for r in metric_results
                ],
            )

            metrics = round_dict_values(metrics)

            # -------------------------------------------------
            # Save evaluation result
            # -------------------------------------------------

            if metrics:

                quantitative_result = {
                    "model_name": model_spec,

                    "task": task,

                    "ethical_principle": principle,

                    "technical_requirement": requirement,

                    # Keep for backwards compatibility
                    "category": requirement,

                    "metrics": metrics,

                    "metric_results": metric_results,

                    "log_dir": log_dir,
                }

                saved_path = save_evaluation_result(
                    model_name=model_spec,
                    evaluation_name=f"quantitative_{task}",
                    result=quantitative_result,
                )

                st.success(
                    f"Saved quantitative result to {saved_path}"
                )

                # ---------------------------------------------
                # Display metrics
                # ---------------------------------------------

                metric_items = list(
                    metrics.items()
                )

                n_cols = min(
                    3,
                    len(metric_items),
                )

                cols = metrics_area.columns(
                    n_cols
                )

                for i, (name, value) in enumerate(
                    metric_items
                ):

                    cols[
                        i % n_cols
                    ].metric(
                        prettify_metric_name(name),
                        f"{value:.2f}",
                    )

                # ---------------------------------------------
                # Rebuild report
                # ---------------------------------------------

                saved_results = load_model_results(
                    model_spec
                )

                report_to_render = (
                    build_model_report_from_saved_results(
                        saved_results
                    )
                )

            else:

                metrics_area.info(
                    "No metrics found in logs.json"
                )

            # -------------------------------------------------
            # Evaluation status
            # -------------------------------------------------

            if process.returncode == 0:

                st.success(
                    "✅ Evaluation finished."
                )

            else:

                st.error(
                    "❌ Evaluation failed"
                )

        except Exception as e:

            st.error(
                f"❌ Failed to run command: {e}"
            )

            if process is not None:
                kill_process_tree(process)

        finally:

            clear_running_evaluation_pid(
                process
            )

            st.session_state.pop(
                "running_eval_process",
                None,
            )

    # ---------------------------------------------------------
    # Model report
    # ---------------------------------------------------------

    with report_placeholder.container():

        render_model_report_panel(
            report_to_render,
            model_spec,
            key_prefix=(
                f"quantitative_report_"
                f"{safe_widget_key(model_spec)}"
            ),
            show_reset=True,
        )


# ============================================================
# Streamlit application
# ============================================================

st.set_page_config(
    page_title="Ensemble evaluation platform",
    layout="wide",
)

render_header()

evaluation_mode = st.sidebar.radio(
    "Evaluation Mode",
    [
        "Quantitative Benchmarks",
        "Qualitative Evaluation",
    ],
)

if evaluation_mode == "Quantitative Benchmarks":

    render_quantitative_benchmark_page()

else:

    render_altai_qualitative_page()