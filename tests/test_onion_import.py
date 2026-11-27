"""Nhập Music4All-Onion (last.fm) từ Zenodo thành parquet (ticket #54): chọn cột Essentia, đổi lượt nghe, luồng tải + md5."""
import bz2
import hashlib
import json

import pandas as pd
import pytest

from src.data.onion import ESSENTIA_FILE, EVENTS_FILE, HPCP_MEAN, README_FILE, convert_events, essentia_columns, run_import
from src.data.schema import SchemaError

WANTED = [
    "id", "rhythm.bpm", "rhythm.bpm_histogram_first_peak_bpm", "rhythm.bpm_histogram_second_peak_bpm", "rhythm.onset_rate",
    "rhythm.danceability", "tonal.key_edma.strength", "tonal.key_krumhansl.strength", "tonal.key_temperley.strength",
    "lowlevel.loudness_ebu128.integrated", "lowlevel.average_loudness",
]
HEADER = WANTED[::-1] + [f"tonal.hpcp.mean_{i}" for i in range(36)] + ["tonal.hpcp.stdev_0", "tonal.thpcp_0", "rhythm.bpm_histogram_0"]


def test_essentia_columns_keep_the_wanted_ones_plus_36_mean_hpcp_bins_in_bin_order():
    cols = essentia_columns(HEADER)
    assert cols[: len(WANTED)] == WANTED
    assert cols[len(WANTED) :] == [f"tonal.hpcp.mean_{i}" for i in range(36)] == list(HPCP_MEAN)
    assert "tonal.hpcp.stdev_0" not in cols and "tonal.thpcp_0" not in cols


def test_a_missing_essentia_column_is_named_in_the_error():
    with pytest.raises(SchemaError, match="tonal.hpcp.mean_7.*rhythm.onset_rate|rhythm.onset_rate.*tonal.hpcp.mean_7"):
        essentia_columns([c for c in HEADER if c not in ("rhythm.onset_rate", "tonal.hpcp.mean_7")])


EVENTS = (
    "user_id\ttrack_id\ttimestamp\r\n"
    "51549\tiJTBIGHPjgJcT4Bt\t2013-01-27 21:42:38\r\n"
    "51549\tiJTBIGHPjgJcT4Bt\t2013-01-27 21:38:53\r\n"
    "7\tAAAAAAAAAAAAAAAA\t2005-02-14 00:00:01\r\n"
    "7\tiJTBIGHPjgJcT4Bt\t2019-12-31 23:59:59\r\n"
    "51549\tAAAAAAAAAAAAAAAA\t2013-01-27 21:42:38\r\n"  # trùng thời điểm với dòng 1: vẫn giữ
)


def test_listening_events_convert_chunk_by_chunk_keeping_every_row_with_typed_columns(tmp_path):
    src = tmp_path / "userid_trackid_timestamp.tsv.bz2"
    src.write_bytes(bz2.compress(EVENTS.encode()))
    stats = convert_events(src, tmp_path / "listening_events.parquet", chunksize=2)

    out = pd.read_parquet(tmp_path / "listening_events.parquet")
    assert len(out) == 5 and out["user_id"].tolist() == [51549, 51549, 7, 7, 51549]  # giữ thứ tự và mọi dòng
    assert str(out["user_id"].dtype) == "int32" and out["track_id"].tolist()[2] == "AAAAAAAAAAAAAAAA"
    assert out["timestamp"].iloc[0] == pd.Timestamp("2013-01-27 21:42:38")
    assert stats == {
        "rows": 5, "users": 2, "tracks": 2,
        "timestamp_min": "2005-02-14 00:00:01", "timestamp_max": "2019-12-31 23:59:59",
    }


def essentia_tsv() -> str:
    header = HEADER + ["lowlevel.barkbands.mean_0"]
    rows = []
    for track, bpm in (("AAAAAAAAAAAAAAAA", 120.5), ("iJTBIGHPjgJcT4Bt", 98.0)):
        values = {c: "0.25" for c in header}
        values["id"], values["rhythm.bpm"] = track, str(bpm)
        rows.append("\t".join(values[c] for c in header))
    return "\r\n".join(["\t".join(header), *rows]) + "\r\n"


@pytest.fixture
def zenodo(tmp_path):
    """Ba file nguồn tự dựng, phục vụ qua URL file://, kèm md5 đúng của hai file dữ liệu."""
    src = tmp_path / "zenodo"
    src.mkdir()
    (src / EVENTS_FILE).write_bytes(bz2.compress(EVENTS.encode()))
    (src / ESSENTIA_FILE).write_bytes(bz2.compress(essentia_tsv().encode()))
    (src / README_FILE).write_text("# Music4All-Onion" + chr(10) + "CC BY 4.0" + chr(10), encoding="utf-8")
    data_md5 = {name: hashlib.md5((src / name).read_bytes()).hexdigest() for name in (EVENTS_FILE, ESSENTIA_FILE)}
    return src, data_md5


# Số liệu đúng của dữ liệu tự dựng: 5 lượt nghe, 2 user, 2 bài Essentia
FAKE_EXPECTED = {"events_rows": 5, "events_users": 2, "essentia_tracks": 2}


def run_fake(src, tmp_path, **overrides):
    kwargs = {"url_for": lambda name: (src / name).as_uri(), "expected": FAKE_EXPECTED, **overrides}
    return run_import(tmp_path / "work", tmp_path / "out" / "onion", **kwargs)


def test_import_downloads_checks_md5_converts_and_logs(zenodo, tmp_path):
    src, data_md5 = zenodo
    log = run_fake(src, tmp_path, data_md5=data_md5)

    out = tmp_path / "out" / "onion"
    events = pd.read_parquet(out / "listening_events.parquet")
    essentia = pd.read_parquet(out / "essentia.parquet")
    assert len(events) == 5
    assert essentia.columns.tolist() == WANTED + list(HPCP_MEAN)  # bỏ cột không cần (barkbands, stdev, thpcp)
    assert essentia["rhythm.bpm"].tolist() == pytest.approx([120.5, 98.0])
    assert (out / "README.md").read_text(encoding="utf-8").startswith("# Music4All-Onion")

    assert json.loads((out / "import_log.json").read_text(encoding="utf-8")) == log
    assert {name: log["md5"][name] for name in data_md5} == data_md5
    assert log["md5"][README_FILE] == hashlib.md5((src / README_FILE).read_bytes()).hexdigest()  # ghi lại, không chặn
    assert log["events"]["rows"] == 5 and log["events"]["users"] == 2
    assert log["essentia"] == {"rows": 2, "tracks": 2, "columns": len(WANTED) + 36}
    assert all(check["ok"] for check in log["check"].values())


def test_a_wrong_data_md5_stops_the_import_before_writing_any_output(zenodo, tmp_path):
    src, data_md5 = zenodo
    with pytest.raises(ValueError, match=f"md5.*{ESSENTIA_FILE}|{ESSENTIA_FILE}.*md5"):
        run_fake(src, tmp_path, data_md5={**data_md5, ESSENTIA_FILE: "0" * 32})
    out = tmp_path / "out" / "onion"
    assert not out.exists() or not any(out.iterdir())


def test_a_row_count_that_differs_from_the_zenodo_readme_fails_without_a_log(zenodo, tmp_path):
    src, data_md5 = zenodo
    with pytest.raises(ValueError, match="events_rows"):
        run_fake(src, tmp_path, data_md5=data_md5, expected={**FAKE_EXPECTED, "events_rows": 6})
    assert not (tmp_path / "out" / "onion" / "import_log.json").exists()


def test_differing_user_or_essentia_track_counts_are_logged_but_do_not_block(zenodo, tmp_path):
    # README của record để trống số bài Essentia (thực tế 109.180, các bộ khác 109.269): chỉ so sánh, không chặn
    src, data_md5 = zenodo
    log = run_fake(src, tmp_path, data_md5=data_md5, expected={**FAKE_EXPECTED, "events_users": 3, "essentia_tracks": 3})
    assert log["check"]["events_users"] == {"expected": 3, "actual": 2, "ok": False}
    assert log["check"]["essentia_tracks"] == {"expected": 3, "actual": 2, "ok": False}
    assert (tmp_path / "out" / "onion" / "import_log.json").exists()
