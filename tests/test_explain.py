"""Dòng giải thích cho một lần chuyển bài trong hàng đợi (ticket #21, spec §8 mục 2)."""
import pytest

from src.rerank.explain import explain


def track(bpm: float, camelot: str, energy: float) -> dict:
    return {"bpm": bpm, "camelot": camelot, "energy": energy}


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # cùng mã Camelot
        (track(124, "8A", 0.71), track(126, "8A", 0.68), "cùng tông 8A · BPM 124→126 · năng lượng 0,71→0,68"),
        # tông kề trên vòng (±1 số cùng chữ, hoặc cùng số khác chữ): mix được
        (track(124, "8A", 0.5), track(124, "9A", 0.5), "tông kề 8A→9A · BPM 124→124 · năng lượng 0,50→0,50"),
        (track(124, "8A", 0.5), track(124, "8B", 0.5), "tông kề 8A→8B · BPM 124→124 · năng lượng 0,50→0,50"),
        # không tương thích
        (track(124, "8A", 0.5), track(124, "3B", 0.5), "đổi tông 8A→3B · BPM 124→124 · năng lượng 0,50→0,50"),
        # BPM khớp nhờ gấp đôi / một nửa (luật xét cả ×2 và ×½)
        (track(70, "8A", 0.5), track(141, "8A", 0.5), "cùng tông 8A · BPM 70→141 (nhịp ×2) · năng lượng 0,50→0,50"),
        (track(140, "8A", 0.5), track(71, "8A", 0.5), "cùng tông 8A · BPM 140→71 (nhịp ½) · năng lượng 0,50→0,50"),
        # ×2 gần hơn mức thường nhưng vẫn lệch xa (214 so với 152: 29% > 12%, luật cho 0 điểm BPM): không ghi nhãn
        (track(107, "3A", 0.5), track(152, "3A", 0.5), "cùng tông 3A · BPM 107→152 · năng lượng 0,50→0,50"),
        (track(146, "8A", 0.5), track(92, "8A", 0.5), "cùng tông 8A · BPM 146→92 · năng lượng 0,50→0,50"),
        # BPM làm tròn để đọc; 99,6 → 100
        (track(99.6, "12B", 0.333), track(101.2, "1B", 0.9), "tông kề 12B→1B · BPM 100→101 · năng lượng 0,33→0,90"),
    ],
)
def test_explanations_name_the_key_relation_bpm_and_energy(a, b, expected):
    assert explain(a, b) == expected
