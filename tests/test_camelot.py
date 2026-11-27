import pytest

from src.features.camelot import camelot_code, camelot_distance, parse_camelot

PITCH_CLASS = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}

# Vòng Camelot chuẩn (Mixed In Key), viết theo tên nốt để không suy ra từ công thức trong code.
MAJOR = {"C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B", "B": "1B",
         "F#": "2B", "C#": "3B", "G#": "4B", "D#": "5B", "A#": "6B", "F": "7B"}
MINOR = {"A": "8A", "E": "9A", "B": "10A", "F#": "11A", "C#": "12A", "G#": "1A",
         "D#": "2A", "A#": "3A", "F": "4A", "C": "5A", "G": "6A", "D": "7A"}

CASES = [(PITCH_CLASS[n], 1, code) for n, code in MAJOR.items()] + [(PITCH_CLASS[n], 0, code) for n, code in MINOR.items()]


@pytest.mark.parametrize("key, mode, expected", CASES)
def test_camelot_code_matches_the_wheel(key, mode, expected):
    assert camelot_code(key, mode) == expected


def test_all_24_keys_map_to_24_distinct_codes():
    codes = {camelot_code(k, m) for k in range(12) for m in (0, 1)}
    assert len(codes) == 24


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("8A", "8A", 0),
        ("8A", "8B", 1),   # cùng số khác chữ
        ("8A", "9A", 1),
        ("12A", "1A", 1),  # vòng quay qua 12 → 1
        ("8A", "9B", 2),   # bước chéo: 1 bước vòng + đổi chữ
        ("8A", "11A", 3),
        ("8A", "2A", 6),   # xa nhất trên vòng
        ("8A", "2B", 7),   # xa nhất cả vòng lẫn chữ
    ],
)
def test_camelot_distance_counts_wheel_steps_plus_a_letter_change(a, b, expected):
    assert camelot_distance(*parse_camelot([a]), *parse_camelot([b]))[0] == expected
