from pathlib import Path

import pytest

from mimicplay.data.dataset import compute_dataset_stats


def test_compute_dataset_stats_handles_empty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    compute_dataset_stats(tmp_path)
    captured = capsys.readouterr()
    assert "No demo_*.hdf5 files found" in captured.out

