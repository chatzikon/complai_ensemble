from __future__ import annotations

import streamlit as st

from complai._cli.utils import get_log_dir

from config import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TASK,
    LOCAL_PROVIDERS,
    PROVIDER_DEFAULT_MODELS,
)
from utils.devices import default_device_value, discover_device_options
from utils.discovery import discover_tasks
from utils.metrics import extract_metrics_from_log_dir, prettify_metric_name
from utils.runner import stream_command


def build_model_string(provider: str, model_name: str) -> str:
    model_name = model_name.strip()
    if not model_name:
        return provider
    return f"{provider}/{model_name}"


st.set_page_config(page_title="Ensemble evaluation platform", layout="wide")
from pathlib import Path

col_left, col_right = st.columns([6, 1])

with col_left:
    st.title("Evaluation of ENSEMBLE’s AI solutions")
    #st.markdown("Run and evaluate models across COMPL-AI benchmarks")

with col_right:
    logo_path = Path(__file__).parent / "assets" / "ENSEMBLE-logo_VNtF8Gw.png"
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)


task_list, task_to_category, category_to_tasks = discover_tasks()
device_options = discover_device_options()

if not task_list:
    task_list = [DEFAULT_TASK]
    task_to_category = {DEFAULT_TASK: "Unknown"}
    category_to_tasks = {"Unknown": [DEFAULT_TASK]}

grouped_task_options = []
for category, tasks in category_to_tasks.items():
    for t in tasks:
        grouped_task_options.append(f"{category} → {t}")

default_task_display = None
for opt in grouped_task_options:
    if opt.endswith(DEFAULT_TASK):
        default_task_display = opt
        break

device_labels = [label for _, label in device_options]
device_label_to_value = {label: value for value, label in device_options}
default_value = default_device_value(device_options)
default_label = next(label for value, label in device_options if value == default_value)

col1, col2 = st.columns(2)

with col1:
    provider = st.selectbox(
        "Provider",
        LOCAL_PROVIDERS,
        index=LOCAL_PROVIDERS.index(DEFAULT_PROVIDER)
        if DEFAULT_PROVIDER in LOCAL_PROVIDERS
        else 0,
    )

with col2:
    selected_task_display = st.selectbox(
        "Benchmark Task",
        grouped_task_options,
        index=grouped_task_options.index(default_task_display)
        if default_task_display in grouped_task_options
        else 0,
    )

task = selected_task_display.split(" → ", 1)[1]
st.caption(f"📂 Category: {task_to_category.get(task, 'Unknown')}")

default_model_for_provider = PROVIDER_DEFAULT_MODELS.get(provider, DEFAULT_MODEL)

col3, col4 = st.columns(2)

with col3:
    model_name = st.text_input(
        "Model name",
        value=default_model_for_provider,
        help="Examples: baseline_gcn, CMAlign, Qwen/Qwen3-8B",
    )

with col4:
    limit = st.text_input("Sample limit (-l)", "")

col5, col6 = st.columns(2)

with col5:
    debug = st.checkbox("Debug (--debug)", value=True)

with col6:
    selected_device_label = st.selectbox(
        "Device",
        device_labels,
        index=device_labels.index(default_label),
        help="Detected automatically from the current machine",
    )

selected_device = device_label_to_value[selected_device_label]
model_spec = build_model_string(provider, model_name)
log_dir = get_log_dir(model_spec, "logs")

cmd = ["complai", "eval", model_spec, "-t", task, "--log-dir", log_dir]

if limit.strip():
    cmd += ["-l", limit.strip()]

if debug:
    cmd.append("--debug")

if selected_device.strip():
    cmd += ["-M", f"device={selected_device.strip()}"]

st.markdown("### CLI Preview")
st.code(" ".join(cmd), language="bash")
st.caption(f"Run logs will be saved to: `{log_dir}`")

if st.button("▶ Run Evaluation"):
    logs_col, metrics_col = st.columns([1.6, 1])

    with logs_col:
        st.markdown("### 📟 Logs")
        log_area = st.empty()

    with metrics_col:
        st.markdown("### 📊 Metrics")
        metrics_area = st.empty()

    logs = ""

    try:
        iterator, process = stream_command(cmd)

        for line in iterator:
            logs += line
            log_area.code(logs)

        process.wait()

        metrics = extract_metrics_from_log_dir(log_dir)

        if metrics:
            metric_items = list(metrics.items())
            n_cols = min(3, len(metric_items))
            cols = metrics_area.columns(n_cols)

            for i, (name, value) in enumerate(metric_items):
                cols[i % n_cols].metric(prettify_metric_name(name), f"{value:.4f}")
        else:
            metrics_area.info("No metrics found in logs.json")

        if process.returncode == 0:
            st.success("✅ Evaluation finished.")
        else:
            st.error("❌ Evaluation failed")

    except Exception as e:
        st.error(f"❌ Failed to run command: {e}")