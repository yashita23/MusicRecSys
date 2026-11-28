"""Đọc app/config.toml: đường dẫn dữ liệu app dùng nằm trong file này (#11; từ #21 là một thư mục artifacts/)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from src.export_artifacts import DEMO, INDEX, MANIFEST, MIX_PARAMS, TRACKS, VECTORS

DEFAULT_CONFIG = Path(__file__).with_name("config.toml")
REQUIRED_FILES = (MANIFEST, TRACKS, VECTORS, INDEX, DEMO, MIX_PARAMS)  # mix_learned.npz là tùy chọn


@dataclass(frozen=True)
class AppConfig:
    """Đường dẫn tuyệt đối tới thư mục artifacts/ (#20); đường dẫn tương đối trong config tính theo thư mục chứa config."""

    artifacts: Path

    @property
    def files(self) -> tuple[Path, ...]:
        """Các file bắt buộc của artifacts/ (để báo thiếu và để nhận biết dữ liệu đổi)."""
        return tuple(self.artifacts / name for name in REQUIRED_FILES)

    def missing_files(self) -> list[Path]:
        return [p for p in self.files if not p.is_file()]


def load_config(path: str | Path = DEFAULT_CONFIG) -> AppConfig:
    """Lỗi FileNotFoundError nếu không có file config, ValueError nếu TOML sai hoặc thiếu đường dẫn."""
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f).get("data", {})
    keys = [field.name for field in fields(AppConfig)]
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"{path}: mục [data] thiếu đường dẫn {', '.join(missing)}")
    return AppConfig(**{key: (path.parent / data[key]).resolve() for key in keys})
