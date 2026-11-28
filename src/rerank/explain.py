"""Một dòng giải thích cho lần chuyển từ bài đang phát a sang bài b (ticket #21, spec §8 mục 2).

Ví dụ: "cùng tông 8A · BPM 124→126 · năng lượng 0,71→0,68". Dùng cùng định nghĩa tông tương thích và BPM xét cả
nhịp gấp đôi/một nửa như điểm mix theo luật (#12), nên lời giải thích khớp với điểm mix hiển thị cạnh nó.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from src.features.camelot import compatible_keys, parse_camelot, same_key
from src.rerank.mix_rule import BPM_ZERO


def _key_phrase(camelot_a: str, camelot_b: str) -> str:
    a, b = parse_camelot([camelot_a]), parse_camelot([camelot_b])
    if same_key(*a, *b)[0]:
        return f"cùng tông {camelot_a}"
    if compatible_keys(*a, *b)[0]:
        return f"tông kề {camelot_a}→{camelot_b}"
    return f"đổi tông {camelot_a}→{camelot_b}"


def _bpm_phrase(bpm_a: float, bpm_b: float) -> str:
    """Ghi chú (nhịp ×2) hoặc (nhịp ½) khi b khớp 2a hoặc a/2 (lệch ≤ BPM_ZERO = 12%, ngưỡng luật còn cho điểm BPM)
    và gần hơn mức lệch thường; xa hơn thì nhân/chia đôi cũng không giúp mix được nên không ghi."""
    plain, doubled, halved = abs(bpm_b - bpm_a) / bpm_a, abs(bpm_b - 2 * bpm_a) / (2 * bpm_a), abs(bpm_b - bpm_a / 2) / (bpm_a / 2)
    note = ""
    if min(doubled, halved) < plain and min(doubled, halved) <= BPM_ZERO:
        note = " (nhịp ×2)" if doubled < halved else " (nhịp ½)"
    return f"BPM {int(np.round(bpm_a))}→{int(np.round(bpm_b))}{note}"


def _decimal(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def explain(a: Mapping, b: Mapping) -> str:
    """a, b có bpm, camelot, energy (một dòng của tracks.parquet)."""
    return " · ".join(
        [
            _key_phrase(str(a["camelot"]), str(b["camelot"])),
            _bpm_phrase(float(a["bpm"]), float(b["bpm"])),
            f"năng lượng {_decimal(float(a['energy']))}→{_decimal(float(b['energy']))}",
        ]
    )
