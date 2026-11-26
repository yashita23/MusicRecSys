"""Đọc app/config.toml: mọi đường dẫn dữ liệu app dùng đều nằm trong file này (#11)."""
from __future__ import annotations

import tomllib
from dataclasses import astuple, dataclass, fields
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("config.toml")


@dataclass(frozen=True)
class AppConfig:
    """Đường dẫn tuyệt đối tới các file app đọc; đường dẫn tương đối trong config tính theo thư mục chứa config."""

    tracks: Path
    eval_targets: Path
    candidates: Path

    @property
    def files(self) -> tuple[Path, ...]:
        return astuple(self)

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
