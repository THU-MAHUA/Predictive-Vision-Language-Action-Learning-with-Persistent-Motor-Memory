import json

import numpy as np
import pandas as pd
import pytest

from conftest import require_torch

require_torch()

from ipme_vla.config import IPMEVLAConfig
from ipme_vla.constants import ACTION, OBS_STATE, TASK
from ipme_vla.data import DatasetFormatError, LeRobotV21Dataset, LeRobotV21Metadata


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def make_fixture(root):
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            OBS_STATE: {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
            ACTION: {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    write_jsonl(root / "meta" / "episodes.jsonl", [{"episode_index": 0, "length": 6, "tasks": [0]}])
    write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "reach"}])
    frame = pd.DataFrame(
        {
            OBS_STATE: [np.asarray([index, index + 1], dtype=np.float32) for index in range(6)],
            ACTION: [np.asarray([index, -index], dtype=np.float32) for index in range(6)],
            "task_index": [0] * 6,
        }
    )
    frame.to_parquet(root / "data" / "chunk-000" / "episode_000000.parquet")


def test_metadata_and_temporal_windows(tmp_path) -> None:
    make_fixture(tmp_path)
    config = IPMEVLAConfig(
        chunk_size=3,
        n_action_steps=1,
        predictive_horizon=1,
        recurrent_unroll_steps=2,
        max_state_dim=4,
        max_action_dim=4,
    )
    dataset = LeRobotV21Dataset(tmp_path, config)
    sample = dataset[4]
    assert sample[OBS_STATE].shape == (3, 2)
    assert sample[ACTION].shape == (4, 2)
    assert sample[TASK] == "reach"
    assert sample["action_is_pad"].tolist() == [False, False, True, True]
    stats = dataset.normalization_stats()
    assert len(stats["state"]["mean"]) == 2


def test_malformed_metadata_has_clear_error(tmp_path) -> None:
    (tmp_path / "meta").mkdir()
    with pytest.raises(DatasetFormatError, match="missing required"):
        LeRobotV21Metadata(tmp_path)
