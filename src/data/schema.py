"""Kiểm tra định dạng file bàn giao theo docs/data-contract.md."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

TRACKS_COLUMNS = {
    "track_id": "string",
    "track_name": "string",
    "artist_name": "string",
    "tempo": "float",
    "bpm": "float",
    "key": "int",
    "mode": "int",
    "camelot": "string",
    "energy": "float",
    "n_occurrences": "int",
}


SESSIONS_COLUMNS = {"session_id": "int", "split": "string", "pos": "int", "track_id": "string"}
SPLITS = ("train", "val", "test")
EVAL_SPLITS = ("val", "test")
MIN_SESSION_LENGTH = 10
SEED_LENGTH = 5

N_CANDIDATES = 100
CANDIDATES_COLUMNS = {"session_id": "int", "rank": "int", "track_id": "string", "taste_score": "float"}

EMBEDDING_INDEX_COLUMNS = {"row": "int", "track_id": "string"}

EVAL_TARGETS_COLUMNS = {
    "session_id": "int",
    "split": "string",
    "seed_track_ids": "list",
    "target_track_id": "string",
}


class SchemaError(ValueError):
    """Dữ liệu không đúng định dạng; thông điệp liệt kê mọi lỗi tìm thấy."""

    def __init__(self, table: str, problems: list[str]):
        self.table = table
        self.problems = problems
        super().__init__(f"{table}: " + "; ".join(problems))


TRACK_ID_PATTERN = r"^[A-Za-z0-9]{22}$"
CAMELOT_PATTERN = r"^(?:[1-9]|1[0-2])[AB]$"


def _missing_columns(df: pd.DataFrame, columns: dict[str, str]) -> list[str]:
    return [f"thiếu cột {c}" for c in columns if c not in df.columns]


def _type_problems(df: pd.DataFrame, columns: dict[str, str]) -> list[str]:
    """Null và sai kiểu cho các cột đã có. 'float' nhận cả số nguyên; 'int' không nhận số thực."""
    problems = []
    for col, kind in columns.items():
        if col not in df.columns:
            continue
        s = df[col]
        if s.isna().any():
            problems.append(f"cột {col} có {int(s.isna().sum())} giá trị null")
            continue
        ok = {
            "string": lambda: bool(s.map(lambda v: isinstance(v, str)).all()),
            "list": lambda: bool(s.map(lambda v: isinstance(v, (list, tuple, np.ndarray))).all()),
            "int": lambda: pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s),
            "float": lambda: pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s),
        }[kind]()
        if not ok:
            problems.append(f"cột {col} phải có kiểu {kind}, đang là {s.dtype}")
    return problems


def _count_bad(mask: pd.Series) -> int:
    return int(mask.sum())


def _rule(problems: list[str], bad: pd.Series, message: str) -> None:
    n = _count_bad(bad)
    if n:
        problems.append(f"{message} ({n} dòng)")


def _raise_if(table: str, problems: list[str]) -> None:
    if problems:
        raise SchemaError(table, problems)


def _base_problems(df: pd.DataFrame, columns: dict[str, str]) -> list[str]:
    return _missing_columns(df, columns) + _type_problems(df, columns)


def _bad_track_ids(s: pd.Series) -> pd.Series:
    return ~s.astype(str).str.fullmatch(TRACK_ID_PATTERN)


def validate_tracks(df: pd.DataFrame) -> None:
    problems = _base_problems(df, TRACKS_COLUMNS)
    if problems:
        raise SchemaError("tracks", problems)
    _rule(problems, _bad_track_ids(df["track_id"]), "track_id phải là Spotify ID 22 ký tự")
    _rule(problems, df["track_id"].duplicated(), "track_id bị trùng")
    _rule(problems, df["tempo"] <= 0, "tempo phải > 0")
    _rule(problems, df["bpm"] <= 0, "bpm phải > 0")
    _rule(problems, ~df["key"].between(0, 11), "key phải trong 0–11")
    _rule(problems, ~df["mode"].isin([0, 1]), "mode phải là 0 hoặc 1")
    _rule(problems, ~df["energy"].between(0, 1), "energy phải trong 0–1")
    _rule(problems, df["n_occurrences"] < 0, "n_occurrences phải ≥ 0")
    camelot = df["camelot"].astype(str)
    _rule(problems, ~camelot.str.fullmatch(CAMELOT_PATTERN), "camelot phải trong 1A–12B")
    letter_for_mode = df["mode"].map({0: "A", 1: "B"})
    _rule(problems, camelot.str[-1] != letter_for_mode, "camelot phải có chữ A khi mode = 0, chữ B khi mode = 1")
    _raise_if("tracks", problems)


def validate_sessions(df: pd.DataFrame) -> None:
    problems = _base_problems(df, SESSIONS_COLUMNS)
    if problems:
        raise SchemaError("sessions", problems)
    _rule(problems, ~df["split"].isin(SPLITS), f"split phải thuộc {SPLITS}")
    _rule(problems, _bad_track_ids(df["track_id"]), "track_id phải là Spotify ID 22 ký tự")
    _rule(problems, df.duplicated(["session_id", "pos"]), "(session_id, pos) bị trùng")

    g = df.groupby("session_id")
    n_splits = g["split"].nunique()
    _rule(problems, n_splits > 1, "mỗi phiên chỉ được thuộc một split")
    length = g.size()
    _rule(problems, length < MIN_SESSION_LENGTH, f"mỗi phiên phải có ≥ {MIN_SESSION_LENGTH} bài")
    contiguous = (g["pos"].min() == 0) & (g["pos"].max() == length - 1) & (g["pos"].nunique() == length)
    _rule(problems, ~contiguous, "pos phải liền mạch 0…n−1 trong mỗi phiên")
    _raise_if("sessions", problems)


def _is_track_id(v: object) -> bool:
    return isinstance(v, str) and re.fullmatch(TRACK_ID_PATTERN, v) is not None


def validate_eval_targets(df: pd.DataFrame) -> None:
    problems = _base_problems(df, EVAL_TARGETS_COLUMNS)
    if problems:
        raise SchemaError("eval_targets", problems)
    seeds = df["seed_track_ids"].map(list)
    _rule(problems, df["session_id"].duplicated(), "session_id bị trùng")
    _rule(problems, ~df["split"].isin(EVAL_SPLITS), f"split phải thuộc {EVAL_SPLITS}")
    _rule(problems, seeds.map(len) != SEED_LENGTH, f"seed phải có đúng {SEED_LENGTH} bài")
    _rule(problems, ~seeds.map(lambda s: all(_is_track_id(t) for t in s)), "seed_track_ids phải là các Spotify ID 22 ký tự")
    _rule(problems, _bad_track_ids(df["target_track_id"]), "target_track_id phải là Spotify ID 22 ký tự")
    in_seed = pd.Series([t in s for t, s in zip(df["target_track_id"], seeds)], index=df.index)
    _rule(problems, in_seed, "bài đích không được nằm trong seed")
    _raise_if("eval_targets", problems)


def validate_candidates(df: pd.DataFrame, k: int = N_CANDIDATES) -> None:
    problems = _base_problems(df, CANDIDATES_COLUMNS)
    if problems:
        raise SchemaError("candidates", problems)
    _rule(problems, _bad_track_ids(df["track_id"]), "track_id phải là Spotify ID 22 ký tự")
    _rule(problems, df.duplicated(["session_id", "track_id"]), "track_id bị trùng trong cùng một phiên")
    _rule(problems, ~np.isfinite(df["taste_score"].to_numpy(dtype=float)), "taste_score phải hữu hạn")

    g = df.groupby("session_id")
    _rule(problems, g.size() != k, f"mỗi phiên phải có đúng {k} ứng viên")
    ranks_ok = g["rank"].apply(lambda r: sorted(r) == list(range(1, len(r) + 1)))
    _rule(problems, ~ranks_ok, "rank phải liền mạch 1…k trong mỗi phiên")
    by_rank = df.sort_values(["session_id", "rank"])
    increasing = by_rank.groupby("session_id")["taste_score"].diff() > 0
    _rule(problems, increasing, "taste_score không được tăng theo rank")
    _raise_if("candidates", problems)


def validate_item_embeddings(vectors: np.ndarray, index: pd.DataFrame) -> None:
    problems = _base_problems(index, EMBEDDING_INDEX_COLUMNS)
    if vectors.ndim != 2:
        problems.append(f"vector phải là mảng 2 chiều (n, d), đang có shape {vectors.shape}")
    if vectors.dtype != np.float32:
        problems.append(f"vector phải có dtype float32, đang là {vectors.dtype}")
    if problems:
        raise SchemaError("item_embeddings", problems)
    if len(index) != vectors.shape[0]:
        problems.append(f"index có {len(index)} dòng nhưng có {vectors.shape[0]} vector")
    if sorted(index["row"]) != list(range(len(index))):
        problems.append("row phải liền mạch 0…n−1")
    _rule(problems, _bad_track_ids(index["track_id"]), "track_id phải là Spotify ID 22 ký tự")
    _rule(problems, index["track_id"].duplicated(), "track_id bị trùng")
    _raise_if("item_embeddings", problems)


def validate_dataset(
    tracks: pd.DataFrame,
    sessions: pd.DataFrame,
    eval_targets: pd.DataFrame,
    candidates: dict[str, pd.DataFrame] | None = None,
    embeddings: dict[str, tuple[np.ndarray, pd.DataFrame]] | None = None,
    k: int = N_CANDIDATES,
) -> None:
    """Kiểm tra từng bảng rồi kiểm tra các bảng khớp nhau.

    candidates: {split: bảng candidates của split đó}. embeddings: {model: (vectors, index)}.
    """
    candidates = candidates or {}
    embeddings = embeddings or {}
    validate_tracks(tracks)
    validate_sessions(sessions)
    validate_eval_targets(eval_targets)
    for cands in candidates.values():
        validate_candidates(cands, k=k)
    for vectors, index in embeddings.values():
        validate_item_embeddings(vectors, index)

    problems: list[str] = []
    known = set(tracks["track_id"])
    _rule(problems, ~sessions["track_id"].isin(known), "sessions có track_id không có trong tracks")

    train_counts = sessions.loc[sessions["split"] == "train", "track_id"].value_counts()
    expected = tracks["track_id"].map(train_counts).fillna(0).astype(int)
    _rule(problems, tracks["n_occurrences"] != expected, "n_occurrences phải bằng số lượt xuất hiện trong các phiên train")

    by_session = sessions.sort_values(["session_id", "pos"]).groupby("session_id")
    session_split = by_session["split"].first()
    session_tracks = by_session["track_id"].agg(list)
    et = eval_targets.set_index("session_id")
    _rule(problems, ~et.index.isin(session_split.index), "eval_targets có session_id không có trong sessions")
    et = et[et.index.isin(session_split.index)]
    _rule(problems, et["split"] != session_split.reindex(et.index), "split của eval_targets phải trùng split của phiên")
    first = session_tracks.reindex(et.index)
    seed_ok = [list(s) == t[:SEED_LENGTH] for s, t in zip(et["seed_track_ids"], first)]
    _rule(problems, ~pd.Series(seed_ok, index=et.index), f"seed phải là {SEED_LENGTH} bài đầu của phiên (pos 0–{SEED_LENGTH - 1})")
    target_ok = [len(t) > SEED_LENGTH and tgt == t[SEED_LENGTH] for tgt, t in zip(et["target_track_id"], first)]
    _rule(problems, ~pd.Series(target_ok, index=et.index), f"bài đích phải là bài ở pos {SEED_LENGTH} của phiên")

    for split, cands in candidates.items():
        in_split = set(eval_targets.loc[eval_targets["split"] == split, "session_id"])
        _rule(problems, ~cands["session_id"].isin(in_split), f"candidates[{split}] có phiên không có trong eval_targets của {split}")
        _rule(problems, ~cands["track_id"].isin(known), f"candidates[{split}] có track_id không có trong tracks")
        seeds = eval_targets.set_index("session_id")["seed_track_ids"].map(set)
        seed_of_row = cands["session_id"].map(seeds)
        in_seed = [isinstance(s, set) and t in s for t, s in zip(cands["track_id"], seed_of_row)]
        _rule(problems, pd.Series(in_seed, index=cands.index), f"candidates[{split}] chứa bài trong seed")

    for model, (_, index) in embeddings.items():
        _rule(problems, ~index["track_id"].isin(known), f"item_embeddings[{model}] có track_id không có trong tracks")
    _raise_if("dataset", problems)
