"""Kiểm tra định dạng file bàn giao theo docs/data-contract.md."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.camelot import CAMELOT_PATTERN, camelot_code

SPLITS = ("train", "val", "test")
EVAL_SPLITS = ("val", "test")
MIN_SESSION_LENGTH = 10
SEED_LENGTH = 5
N_CANDIDATES = 100

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
EVAL_TARGETS_COLUMNS = {"session_id": "int", "split": "string", "seed_track_ids": "list", "target_track_id": "string"}
CANDIDATES_COLUMNS = {"session_id": "int", "rank": "int", "track_id": "string", "taste_score": "float"}
EMBEDDING_INDEX_COLUMNS = {"row": "int", "track_id": "string"}

TRACK_ID_PATTERN = r"^[A-Za-z0-9]{22}$"
NOT_A_TRACK_ID = "phải là Spotify ID 22 ký tự"


class SchemaError(ValueError):
    """Dữ liệu không đúng định dạng; thông điệp liệt kê mọi lỗi tìm thấy."""

    def __init__(self, table: str, problems: list[str]):
        self.table = table
        self.problems = problems
        super().__init__(f"{table}: " + "; ".join(problems))


# ---------- tiện ích dùng chung với pipeline và fake.py ----------

def count_train_occurrences(sessions: pd.DataFrame) -> pd.Series:
    """Số lượt xuất hiện của mỗi bài trong các phiên train (định nghĩa của n_occurrences)."""
    return sessions.loc[sessions["split"] == "train", "track_id"].value_counts()


def session_track_lists(sessions: pd.DataFrame) -> pd.Series:
    """session_id → danh sách track_id theo thứ tự pos."""
    return sessions.sort_values(["session_id", "pos"]).groupby("session_id")["track_id"].agg(list)


# ---------- ghi nhận lỗi ----------

def _flag(problems: list[str], bad: pd.Series, message: str) -> None:
    """Ghi một lỗi nếu có dòng nào vi phạm."""
    n = int(bad.sum())
    if n:
        problems.append(f"{message} ({n} dòng)")


def _raise_if_any(table: str, problems: list[str]) -> None:
    if problems:
        raise SchemaError(table, problems)


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
        if kind == "string":
            ok = bool(s.map(lambda v: isinstance(v, str)).all())
        elif kind == "list":
            ok = bool(s.map(lambda v: isinstance(v, (list, tuple, np.ndarray))).all())
        else:
            numeric = pd.api.types.is_integer_dtype(s) if kind == "int" else pd.api.types.is_numeric_dtype(s)
            ok = numeric and not pd.api.types.is_bool_dtype(s)
        if not ok:
            problems.append(f"cột {col} phải có kiểu {kind}, đang là {s.dtype}")
    return problems


def _require_columns(table: str, df: pd.DataFrame, columns: dict[str, str]) -> list[str]:
    """Dừng sớm nếu thiếu cột hoặc sai kiểu (các luật sau cần cột đúng kiểu); trả về danh sách lỗi rỗng."""
    problems = [f"thiếu cột {c}" for c in columns if c not in df.columns] + _type_problems(df, columns)
    _raise_if_any(table, problems)
    return problems


def _bad_track_ids(s: pd.Series) -> pd.Series:
    return ~s.astype(str).str.fullmatch(TRACK_ID_PATTERN)


# ---------- từng bảng ----------

def validate_tracks(df: pd.DataFrame) -> None:
    problems = _require_columns("tracks", df, TRACKS_COLUMNS)
    _flag(problems, _bad_track_ids(df["track_id"]), f"track_id {NOT_A_TRACK_ID}")
    _flag(problems, df["track_id"].duplicated(), "track_id bị trùng")
    _flag(problems, df["tempo"] <= 0, "tempo phải > 0")
    _flag(problems, df["bpm"] <= 0, "bpm phải > 0")
    _flag(problems, ~df["key"].between(0, 11), "key phải trong 0–11")
    _flag(problems, ~df["mode"].isin([0, 1]), "mode phải là 0 hoặc 1")
    _flag(problems, ~df["energy"].between(0, 1), "energy phải trong 0–1")
    _flag(problems, df["n_occurrences"] < 0, "n_occurrences phải ≥ 0")
    camelot = df["camelot"].astype(str)
    _flag(problems, ~camelot.str.fullmatch(CAMELOT_PATTERN), "camelot phải trong 1A–12B")
    expected = pd.Series([camelot_code(int(k), int(m)) for k, m in zip(df["key"], df["mode"])], index=df.index)
    _flag(problems, camelot != expected, "camelot phải khớp key và mode (C trưởng = 8B, A thứ = 8A)")
    _raise_if_any("tracks", problems)


def validate_sessions(df: pd.DataFrame) -> None:
    problems = _require_columns("sessions", df, SESSIONS_COLUMNS)
    _flag(problems, ~df["split"].isin(SPLITS), f"split phải thuộc {SPLITS}")
    _flag(problems, _bad_track_ids(df["track_id"]), f"track_id {NOT_A_TRACK_ID}")
    _flag(problems, df.duplicated(["session_id", "pos"]), "(session_id, pos) bị trùng")
    by_session = df.groupby("session_id")
    _flag(problems, by_session["split"].nunique() > 1, "mỗi phiên chỉ được thuộc một split")
    length = by_session.size()
    _flag(problems, length < MIN_SESSION_LENGTH, f"mỗi phiên phải có ≥ {MIN_SESSION_LENGTH} bài")
    pos = by_session["pos"]
    contiguous = (pos.min() == 0) & (pos.max() == length - 1) & (pos.nunique() == length)
    _flag(problems, ~contiguous, "pos phải liền mạch 0…n−1 trong mỗi phiên")
    _raise_if_any("sessions", problems)


def validate_eval_targets(df: pd.DataFrame) -> None:
    problems = _require_columns("eval_targets", df, EVAL_TARGETS_COLUMNS)
    seeds = df["seed_track_ids"].map(list)
    _flag(problems, df["session_id"].duplicated(), "session_id bị trùng")
    _flag(problems, ~df["split"].isin(EVAL_SPLITS), f"split phải thuộc {EVAL_SPLITS}")
    _flag(problems, seeds.map(len) != SEED_LENGTH, f"seed phải có đúng {SEED_LENGTH} bài")
    bad_seed_ids = _bad_track_ids(seeds.explode()).groupby(level=0).any().reindex(df.index, fill_value=False)
    _flag(problems, bad_seed_ids, f"seed_track_ids {NOT_A_TRACK_ID}")
    _flag(problems, _bad_track_ids(df["target_track_id"]), f"target_track_id {NOT_A_TRACK_ID}")
    in_seed = pd.Series([t in s for t, s in zip(df["target_track_id"], seeds)], index=df.index)
    _flag(problems, in_seed, "bài đích không được nằm trong seed")
    _raise_if_any("eval_targets", problems)


def validate_candidates(df: pd.DataFrame, k: int = N_CANDIDATES) -> None:
    problems = _require_columns("candidates", df, CANDIDATES_COLUMNS)
    _flag(problems, _bad_track_ids(df["track_id"]), f"track_id {NOT_A_TRACK_ID}")
    _flag(problems, df.duplicated(["session_id", "track_id"]), "track_id bị trùng trong cùng một phiên")
    _flag(problems, ~np.isfinite(df["taste_score"].to_numpy(dtype=float)), "taste_score phải hữu hạn")
    by_session = df.groupby("session_id")
    _flag(problems, by_session.size() != k, f"mỗi phiên phải có đúng {k} ứng viên")
    ranks_ok = by_session["rank"].apply(lambda r: sorted(r) == list(range(1, len(r) + 1)))
    _flag(problems, ~ranks_ok, "rank phải liền mạch 1…k trong mỗi phiên")
    by_rank = df.sort_values(["session_id", "rank"])
    _flag(problems, by_rank.groupby("session_id")["taste_score"].diff() > 0, "taste_score không được tăng theo rank")
    _raise_if_any("candidates", problems)


def validate_item_embeddings(vectors: np.ndarray, index: pd.DataFrame) -> None:
    problems = _require_columns("item_embeddings", index, EMBEDDING_INDEX_COLUMNS)
    if vectors.ndim != 2:
        problems.append(f"vector phải là mảng 2 chiều (n, d), đang có shape {vectors.shape}")
    if vectors.dtype != np.float32:
        problems.append(f"vector phải có dtype float32, đang là {vectors.dtype}")
    _raise_if_any("item_embeddings", problems)
    if not np.isfinite(vectors).all():
        problems.append("mọi phần tử vector phải hữu hạn (không NaN/inf)")
    if len(index) != vectors.shape[0]:
        problems.append(f"index có {len(index)} dòng nhưng có {vectors.shape[0]} vector")
    if sorted(index["row"]) != list(range(len(index))):
        problems.append("row phải liền mạch 0…n−1")
    _flag(problems, _bad_track_ids(index["track_id"]), f"track_id {NOT_A_TRACK_ID}")
    _flag(problems, index["track_id"].duplicated(), "track_id bị trùng")
    _raise_if_any("item_embeddings", problems)


# ---------- liên bảng ----------

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
    _flag(problems, ~sessions["track_id"].isin(known), "sessions có track_id không có trong tracks")
    train_counts = count_train_occurrences(sessions)
    expected = tracks["track_id"].map(train_counts).fillna(0).astype(int)
    _flag(problems, tracks["n_occurrences"] != expected, "n_occurrences phải bằng số lượt xuất hiện trong các phiên train")

    _eval_target_problems(problems, sessions, eval_targets)

    seeds_by_session = eval_targets.set_index("session_id")["seed_track_ids"].map(set)
    for split, cands in candidates.items():
        in_split = set(eval_targets.loc[eval_targets["split"] == split, "session_id"])
        _flag(problems, ~cands["session_id"].isin(in_split), f"candidates[{split}] có phiên không có trong eval_targets của {split}")
        n_without = len(in_split - set(cands["session_id"]))
        if n_without:
            problems.append(f"{n_without} phiên trong eval_targets của {split} không có ứng viên")
        _flag(problems, ~cands["track_id"].isin(known), f"candidates[{split}] có track_id không có trong tracks")
        seed_of_row = cands["session_id"].map(seeds_by_session)
        in_seed = [isinstance(s, set) and t in s for t, s in zip(cands["track_id"], seed_of_row)]
        _flag(problems, pd.Series(in_seed, index=cands.index), f"candidates[{split}] chứa bài trong seed")

    for model, (_, index) in embeddings.items():
        _flag(problems, ~index["track_id"].isin(known), f"item_embeddings[{model}] có track_id không có trong tracks")
        _flag(problems, ~index["track_id"].isin(set(train_counts.index)), f"item_embeddings[{model}] có bài không xuất hiện trong train")
    _raise_if_any("dataset", problems)


def _eval_target_problems(problems: list[str], sessions: pd.DataFrame, eval_targets: pd.DataFrame) -> None:
    """eval_targets khớp với phiên: đúng split, seed = pos 0–4, bài đích = pos 5, không thiếu phiên nào."""
    split_of = sessions.groupby("session_id")["split"].first()
    tracks_of = session_track_lists(sessions)
    targets = eval_targets.set_index("session_id")
    _flag(problems, ~targets.index.isin(split_of.index), "eval_targets có session_id không có trong sessions")
    targets = targets[targets.index.isin(split_of.index)]
    _flag(problems, targets["split"] != split_of.reindex(targets.index), "split của eval_targets phải trùng split của phiên")
    lists = tracks_of.reindex(targets.index)
    seed_ok = [list(s) == t[:SEED_LENGTH] for s, t in zip(targets["seed_track_ids"], lists)]
    _flag(problems, ~pd.Series(seed_ok, index=targets.index), f"seed phải là {SEED_LENGTH} bài đầu của phiên (pos 0–{SEED_LENGTH - 1})")
    target_ok = [len(t) > SEED_LENGTH and tgt == t[SEED_LENGTH] for tgt, t in zip(targets["target_track_id"], lists)]
    _flag(problems, ~pd.Series(target_ok, index=targets.index), f"bài đích phải là bài ở pos {SEED_LENGTH} của phiên")

    eval_sessions = tracks_of[split_of.reindex(tracks_of.index).isin(EVAL_SPLITS)]
    eligible = eval_sessions[eval_sessions.map(lambda t: t[SEED_LENGTH] not in t[:SEED_LENGTH])]
    _flag(problems, ~eligible.index.isin(targets.index), "eval_targets thiếu phiên val/test có bài đích không nằm trong seed")
