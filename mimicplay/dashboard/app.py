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
    from typing import List

    import numpy as np
    import streamlit as st

    from mimicplay.evaluation.evaluator import run_evaluation

    st.set_page_config(page_title="MimicPlay Dashboard", layout="wide")
    st.title("MimicPlay Evaluation Dashboard")

    tab_perf, tab_eval, tab_attention, tab_compare, tab_dataset = st.tabs(
        ["Training Metrics", "Evaluation", "GradCAM Viewer", "Comparison", "Dataset Browser"]
    )

    # Tab 1: Training Metrics
    with tab_perf:
        st.subheader("Training Metrics")
        uploaded = st.file_uploader(
            "Upload training metrics CSV (columns: epoch, train_loss)",
            type=["csv"],
        )
        if uploaded is not None:
            import pandas as pd

            df = pd.read_csv(uploaded)
            if {"epoch", "train_loss"}.issubset(df.columns):
                st.line_chart(df.set_index("epoch")["train_loss"], height=300)
                best_row = df.loc[df["train_loss"].idxmin()]
                col1, col2 = st.columns(2)
                col1.metric("Best Epoch", int(best_row["epoch"]))
                col2.metric("Best Loss", f"{best_row['train_loss']:.4f}")
            else:
                st.warning("CSV must contain 'epoch' and 'train_loss' columns.")
        else:
            st.info("Upload a CSV to visualize training curves.")

    # Tab 2: Evaluation
    with tab_eval:
        st.subheader("Evaluation")
        ckpt_path = st.text_input("Checkpoint path", "checkpoints/bc_grid_collector_best.pt")
        env_name = st.selectbox("Environment", ["grid_collector", "dodge_runner", "build_bridge"])
        task = st.text_input("Task instruction", "collect all coins")
        episodes = st.slider("Episodes", min_value=5, max_value=100, value=20, step=5)
        record_video = st.checkbox("Record MP4 videos", value=False)

        if st.button("Run Evaluation"):
            from io import StringIO
            import sys

            buf = StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            try:
                run_evaluation(
                    checkpoint_path=Path(ckpt_path),
                    env_name=env_name,
                    task_instruction=task,
                    num_episodes=episodes,
                    record_video=record_video,
                )
            except Exception as exc:  # pragma: no cover - dashboard helper
                sys.stdout = old_stdout
                st.error(f"Evaluation failed: {exc}")
            else:
                sys.stdout = old_stdout
                output = buf.getvalue()
                st.text(output)

    # Tab 3: GradCAM Viewer (UI scaffold)
    with tab_attention:
        st.subheader("GradCAM Viewer")
        st.write("Select a checkpoint and demo, then overlay GradCAM on selected frames.")
        st.text_input("Checkpoint path", "checkpoints/bc_grid_collector_best.pt", key="gradcam_ckpt")
        st.text_input("Demo HDF5 path", "demos/grid_collector/collect_all_coins/demo_001.hdf5")
        st.slider("Timestep", min_value=0, max_value=100, value=0)
        st.info("Hook this up to `mimicplay.evaluation.gradcam` to compute real overlays.")

    # Tab 4: Comparison (UI scaffold)
    with tab_compare:
        st.subheader("Model Comparison")
        st.write("Compare multiple checkpoints on the same environment and task.")
        st.text_input("Checkpoint paths (comma-separated)", "checkpoints/bc_grid_collector_best.pt, checkpoints/vla_grid_collector_best.pt")
        st.selectbox("Environment", ["grid_collector", "dodge_runner", "build_bridge"], key="compare_env")
        st.text_input("Task instruction", "collect all coins", key="compare_task")
        st.info("Use the CLI `mimicplay compare` for full comparison; this tab is a lightweight front-end.")

    # Tab 5: Dataset Browser
    with tab_dataset:
        st.subheader("Dataset Browser")
        demo_dir = st.text_input("Demo directory", "demos/grid_collector/collect_all_coins")
        if st.button("Show Stats"):
            from mimicplay.data.dataset import compute_dataset_stats
            from io import StringIO
            import sys

            buf2 = StringIO()
            old_stdout2 = sys.stdout
            sys.stdout = buf2
            try:
                compute_dataset_stats(Path(demo_dir))
            except Exception as exc:  # pragma: no cover
                sys.stdout = old_stdout2
                st.error(f"Failed to read dataset: {exc}")
            else:
                sys.stdout = old_stdout2
                st.text(buf2.getvalue())


if __name__ == "__main__":
    main()

