"""Đọc app/config.toml: mọi đường dẫn dữ liệu app dùng đều nằm trong file này (#11)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("config.toml")
DATA_KEYS = ("tracks", "eval_targets", "candidates")


@dataclass(frozen=True)
class AppConfig:
    """Đường dẫn tuyệt đối tới các file app đọc; đường dẫn tương đối trong config tính theo thư mục chứa config."""

    tracks: Path
    eval_targets: Path
    candidates: Path

    def missing_files(self) -> list[Path]:
        return [p for p in (self.tracks, self.eval_targets, self.candidates) if not p.is_file()]


def load_config(path: str | Path = DEFAULT_CONFIG) -> AppConfig:
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f).get("data", {})
    missing = [key for key in DATA_KEYS if key not in data]
    if missing:
        raise ValueError(f"{path}: mục [data] thiếu đường dẫn {', '.join(missing)}")
    return AppConfig(**{key: (path.parent / data[key]).resolve() for key in DATA_KEYS})
