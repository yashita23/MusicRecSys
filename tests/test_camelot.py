import pytest

from src.features.camelot import camelot_code

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
