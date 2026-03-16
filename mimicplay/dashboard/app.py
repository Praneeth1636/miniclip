"""Streamlit evaluation dashboard."""

from __future__ import annotations

from pathlib import Path


def launch_dashboard() -> None:
    """Launch the Streamlit dashboard as a separate process."""
    import subprocess
    import sys

    app_path = Path(__file__).resolve()
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless=true"]
    subprocess.run(cmd, check=False)


def main() -> None:
    """Entry point for `streamlit run`."""
    import streamlit as st
    import numpy as np

    st.set_page_config(page_title="MimicPlay Dashboard", layout="wide")
    st.title("MimicPlay Evaluation Dashboard")

    tab_perf, tab_replay, tab_attention, tab_compare, tab_dataset = st.tabs(
        ["Performance", "Replay", "Attention", "Comparison", "Dataset"]
    )

    with tab_perf:
        st.subheader("Performance Metrics")
        st.write("Hook this tab into logged metrics (e.g., from wandb or saved JSON).")

    with tab_replay:
        st.subheader("Episode Replays")
        st.write("Display side-by-side human vs agent videos here.")

    with tab_attention:
        st.subheader("GradCAM Attention Maps")
        st.write("Visualize GradCAM overlays for selected frames and actions.")

    with tab_compare:
        st.subheader("Model Comparison")
        st.write("Bar charts comparing BC vs VLA across environments and tasks.")

    with tab_dataset:
        st.subheader("Dataset Browser")
        st.write("List recorded demos, filter by task/success, and show basic stats.")


if __name__ == "__main__":
    main()

