"""Mã Camelot từ key (pitch class 0–11) và mode (0 = thứ, 1 = trưởng)."""
from __future__ import annotations


def camelot_code(key: int, mode: int) -> str:
    """C trưởng = 8B, A thứ = 8A; mỗi bước quãng năm (+7 bán cung) tăng 1 số trên vòng."""
    if mode == 1:
        return f"{(7 * key + 7) % 12 + 1}B"
    return f"{(7 * key + 4) % 12 + 1}A"
