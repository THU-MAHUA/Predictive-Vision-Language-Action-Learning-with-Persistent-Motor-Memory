"""Minimal local LeRobot v2.1 reader for IPME-VLA."""

from __future__ import annotations

import io
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ipme_vla.config import IPMEVLAConfig
from ipme_vla.constants import ACTION, OBS_STATE, TASK


class DatasetFormatError(ValueError):
    """Raised when a local dataset does not satisfy the LeRobot v2.1 contract."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetFormatError(f"Unable to read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise DatasetFormatError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise DatasetFormatError(
                        f"{path}:{line_number} must contain a JSON object"
                    )
                records.append(value)
    except OSError as error:
        raise DatasetFormatError(f"Unable to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise DatasetFormatError(
            f"Malformed JSON in {path} at line {error.lineno}: {error.msg}"
        ) from error
    return records


class LeRobotV21Metadata:
    """Validated metadata and path formatting for a local v2.1 dataset."""

    REQUIRED_FILES = ("info.json", "episodes.jsonl", "tasks.jsonl")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        meta = self.root / "meta"
        missing = [name for name in self.REQUIRED_FILES if not (meta / name).is_file()]
        if missing:
            raise DatasetFormatError(
                f"{self.root} is missing required LeRobot v2.1 metadata: "
                + ", ".join(f"meta/{name}" for name in missing)
            )
        self.info = _read_json(meta / "info.json")
        self.episode_records = _read_jsonl(meta / "episodes.jsonl")
        self.task_records = _read_jsonl(meta / "tasks.jsonl")
        self.episode_stats_records = (
            _read_jsonl(meta / "episodes_stats.jsonl")
            if (meta / "episodes_stats.jsonl").is_file()
            else []
        )
        self._validate()
        self.episodes = {
            int(record["episode_index"]): record for record in self.episode_records
        }
        self.tasks = {
            int(record["task_index"]): str(record["task"]) for record in self.task_records
        }

    def _validate(self) -> None:
        required_info = ("codebase_version", "fps", "chunks_size", "data_path", "features")
        missing = [key for key in required_info if key not in self.info]
        if missing:
            raise DatasetFormatError(
                "meta/info.json is missing required fields: " + ", ".join(missing)
            )
        version = str(self.info["codebase_version"]).lstrip("v")
        if not version.startswith("2.1"):
            raise DatasetFormatError(
                f"IPME-VLA expects LeRobot v2.1 metadata, found {self.info['codebase_version']!r}"
            )
        if not isinstance(self.info["features"], dict):
            raise DatasetFormatError("meta/info.json field 'features' must be an object")
        for key in (OBS_STATE, ACTION):
            feature = self.info["features"].get(key)
            if not isinstance(feature, dict) or not feature.get("shape"):
                raise DatasetFormatError(f"dataset feature {key!r} is required with a shape")
        if not self.episode_records:
            raise DatasetFormatError("meta/episodes.jsonl contains no episodes")
        for record in self.episode_records:
            if "episode_index" not in record or "length" not in record:
                raise DatasetFormatError(
                    "every episodes.jsonl record needs episode_index and length"
                )
            if int(record["length"]) <= 0:
                raise DatasetFormatError("episode lengths must be positive")
        for record in self.task_records:
            if "task_index" not in record or "task" not in record:
                raise DatasetFormatError(
                    "every tasks.jsonl record needs task_index and task"
                )

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        return [
            key
            for key, feature in self.features.items()
            if feature.get("dtype") in {"image", "video"}
        ]

    @property
    def state_dim(self) -> int:
        return int(self.features[OBS_STATE]["shape"][-1])

    @property
    def action_dim(self) -> int:
        return int(self.features[ACTION]["shape"][-1])

    def _format(self, template: str, episode_index: int, **values: Any) -> Path:
        chunk = episode_index // int(self.info["chunks_size"])
        relative = template.format(
            episode_chunk=chunk,
            episode_index=episode_index,
            **values,
        )
        return self.root / relative

    def data_path(self, episode_index: int) -> Path:
        return self._format(str(self.info["data_path"]), episode_index)

    def video_path(self, episode_index: int, video_key: str) -> Path:
        template = self.info.get("video_path")
        if not template:
            raise DatasetFormatError(
                f"feature {video_key!r} is marked as video but info.json has no video_path"
            )
        return self._format(str(template), episode_index, video_key=video_key)

    def configure_model(self, config: IPMEVLAConfig) -> IPMEVLAConfig:
        config.state_dim = self.state_dim
        config.action_dim = self.action_dim
        config.image_features = list(self.image_keys)
        config.__post_init__()
        return config


def _to_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.float()
    if hasattr(value, "as_py"):
        value = value.as_py()
    return torch.from_numpy(np.array(value, dtype=np.float32, copy=True))


def _to_image(value: Any, root: Path | None = None) -> Tensor:
    if isinstance(value, dict) and value.get("bytes") is not None:
        value = Image.open(io.BytesIO(value["bytes"]))
    elif isinstance(value, dict) and value.get("path"):
        path = Path(value["path"])
        value = Image.open(path if path.is_absolute() or root is None else root / path)
    elif isinstance(value, (bytes, bytearray)):
        value = Image.open(io.BytesIO(value))
    elif isinstance(value, str):
        value = Image.open(value)
    if isinstance(value, Image.Image):
        array = np.asarray(value.convert("RGB"))
    else:
        array = np.asarray(value)
    if array.ndim != 3:
        raise DatasetFormatError(f"image value has invalid shape {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        tensor = torch.as_tensor(array[:3], dtype=torch.float32)
    else:
        tensor = torch.as_tensor(array[..., :3], dtype=torch.float32).permute(2, 0, 1)
    if tensor.max() > 1:
        tensor = tensor / 255.0
    return tensor.contiguous()


def _decode_video_frame(path: Path, frame_index: int) -> Tensor:
    try:
        import av
    except ImportError as error:
        raise DatasetFormatError(
            "PyAV is required to decode video-backed LeRobot features"
        ) from error
    if not path.is_file():
        raise DatasetFormatError(f"video file does not exist: {path}")
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if index == frame_index:
                    return _to_image(frame.to_image())
    except Exception as error:
        raise DatasetFormatError(f"unable to decode {path}: {error}") from error
    raise DatasetFormatError(f"frame {frame_index} is outside video {path}")


class LeRobotV21Dataset(Dataset):
    """Episode-local temporal windows needed by recurrent IPME training."""

    def __init__(
        self,
        root: str | Path,
        config: IPMEVLAConfig,
        episode_indices: Iterable[int] | None = None,
        cache_size: int = 4,
    ) -> None:
        self.meta = LeRobotV21Metadata(root)
        self.config = self.meta.configure_model(config)
        selected = (
            sorted(self.meta.episodes)
            if episode_indices is None
            else [int(index) for index in episode_indices]
        )
        unknown = sorted(set(selected) - set(self.meta.episodes))
        if unknown:
            raise DatasetFormatError(f"unknown episode indices: {unknown}")
        self.episode_indices = selected
        self.samples = [
            (episode_index, frame_index)
            for episode_index in selected
            for frame_index in range(int(self.meta.episodes[episode_index]["length"]))
        ]
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[int, pd.DataFrame] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def _episode(self, episode_index: int) -> pd.DataFrame:
        if episode_index in self._cache:
            frame = self._cache.pop(episode_index)
            self._cache[episode_index] = frame
            return frame
        path = self.meta.data_path(episode_index)
        if not path.is_file():
            raise DatasetFormatError(f"episode parquet file does not exist: {path}")
        try:
            frame = pd.read_parquet(path)
        except Exception as error:
            raise DatasetFormatError(f"unable to read episode parquet {path}: {error}") from error
        expected_length = int(self.meta.episodes[episode_index]["length"])
        if len(frame) != expected_length:
            raise DatasetFormatError(
                f"{path} has {len(frame)} rows; metadata declares {expected_length}"
            )
        for key in (OBS_STATE, ACTION):
            if key not in frame.columns:
                raise DatasetFormatError(f"{path} is missing required column {key!r}")
        self._cache[episode_index] = frame
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return frame

    def _indices(
        self, center: int, deltas: Iterable[int], length: int
    ) -> tuple[list[int], Tensor]:
        raw = [center + int(delta) for delta in deltas]
        padding = torch.tensor(
            [index < 0 or index >= length for index in raw], dtype=torch.bool
        )
        return [max(0, min(length - 1, index)) for index in raw], padding

    def _task(self, episode: pd.DataFrame, frame_index: int, episode_index: int) -> str:
        if "task_index" in episode.columns:
            value = episode.iloc[frame_index]["task_index"]
            if isinstance(value, (list, tuple, np.ndarray)):
                value = value[0]
            index = int(value)
            if index in self.meta.tasks:
                return self.meta.tasks[index]
        record = self.meta.episodes[episode_index]
        tasks = record.get("tasks", [])
        if tasks:
            first = tasks[0]
            if isinstance(first, int) and first in self.meta.tasks:
                return self.meta.tasks[first]
            return str(first)
        if len(self.meta.tasks) == 1:
            return next(iter(self.meta.tasks.values()))
        raise DatasetFormatError(
            f"cannot resolve a task for episode {episode_index}, frame {frame_index}"
        )

    def __getitem__(self, item: int) -> dict[str, Any]:
        episode_index, frame_index = self.samples[item]
        episode = self._episode(episode_index)
        length = len(episode)
        observation_indices, observation_padding = self._indices(
            frame_index, self.config.observation_delta_indices, length
        )
        action_indices, action_padding = self._indices(
            frame_index, self.config.action_delta_indices, length
        )
        output: dict[str, Any] = {
            OBS_STATE: torch.stack(
                [_to_tensor(episode.iloc[index][OBS_STATE]) for index in observation_indices]
            ),
            ACTION: torch.stack(
                [_to_tensor(episode.iloc[index][ACTION]) for index in action_indices]
            ),
            f"{OBS_STATE}_is_pad": observation_padding,
            "action_is_pad": action_padding,
            TASK: self._task(episode, frame_index, episode_index),
            "episode_index": episode_index,
            "frame_index": frame_index,
        }
        for key in self.meta.image_keys:
            feature = self.meta.features[key]
            values: list[Tensor] = []
            for index in observation_indices:
                if feature.get("dtype") == "video":
                    values.append(
                        _decode_video_frame(
                            self.meta.video_path(episode_index, key),
                            index,
                        )
                    )
                else:
                    if key not in episode.columns:
                        raise DatasetFormatError(
                            f"image feature {key!r} is absent from episode parquet"
                        )
                    values.append(_to_image(episode.iloc[index][key], self.meta.root))
            output[key] = torch.stack(values)
            output[f"{key}_padding_mask"] = ~observation_padding
        return output

    def normalization_stats(self) -> dict[str, dict[str, list[float]]]:
        """Aggregate mean/std statistics from selected local episode parquet files."""
        state_values: list[np.ndarray] = []
        action_values: list[np.ndarray] = []
        for episode_index in self.episode_indices:
            episode = self._episode(episode_index)
            state_values.extend(np.asarray(value, dtype=np.float32) for value in episode[OBS_STATE])
            action_values.extend(np.asarray(value, dtype=np.float32) for value in episode[ACTION])
        return {
            "state": _mean_std(state_values),
            "action": _mean_std(action_values),
        }


def _mean_std(values: list[np.ndarray]) -> dict[str, list[float]]:
    if not values:
        raise DatasetFormatError("cannot compute statistics from an empty dataset")
    array = np.stack(values)
    return {
        "mean": array.mean(axis=0).tolist(),
        "std": np.maximum(array.std(axis=0), 1e-8).tolist(),
    }


def split_episodes(
    metadata: LeRobotV21Metadata,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    episodes = np.asarray(sorted(metadata.episodes), dtype=np.int64)
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if validation_fraction == 0 or len(episodes) < 2:
        return episodes.tolist(), []
    generator = np.random.default_rng(seed)
    generator.shuffle(episodes)
    count = max(1, round(len(episodes) * validation_fraction))
    return sorted(episodes[count:].tolist()), sorted(episodes[:count].tolist())
