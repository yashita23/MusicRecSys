"""Liệt kê vài ID trong thông báo lỗi."""
from __future__ import annotations

from collections.abc import Iterable

PREVIEW_LIMIT = 5


def preview_ids(ids: Iterable[str]) -> str:
    """Vài ID đầu (đã sắp xếp, bỏ trùng): 'a, b, c, d, e, …'."""
    unique = sorted(set(ids))
    return ", ".join(unique[:PREVIEW_LIMIT]) + (", …" if len(unique) > PREVIEW_LIMIT else "")
