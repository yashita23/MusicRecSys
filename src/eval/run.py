"""Đánh giá độ chính xác của một danh sách xếp hạng so với bài đích (ticket #8).

    python -m src.eval.run --data <thư mục có eval_targets.parquet> --candidates <file candidates>
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.data.schema import SchemaError, validate_candidates
from src.eval.accuracy import MISS, hit_rate, mrr, ndcg

K = 10


@dataclass(frozen=True)
class Accuracy:
    """Độ chính xác trung bình trên các phiên được chấm của một split."""

    split: str
    n_sessions: int
    k: int
    list_length: int  # số bài trong danh sách của mỗi phiên; MRR chỉ so được giữa các lần chấm cùng list_length
    hit_rate: float   # HitRate@k
    ndcg: float       # NDCG@k
    mrr: float        # MRR trên toàn danh sách


def target_ranks(ranked: pd.DataFrame, eval_targets: pd.DataFrame) -> pd.Series:
    """session_id → hạng của bài đích trong danh sách của phiên đó (MISS nếu không có)."""
    target_of = eval_targets.set_index("session_id")["target_track_id"]
    hits = ranked[ranked["track_id"].to_numpy() == ranked["session_id"].map(target_of).to_numpy()]
    return hits.groupby("session_id")["rank"].min().reindex(target_of.index, fill_value=MISS)


def _list_problems(ranked: pd.DataFrame) -> list[str]:
    """Mỗi phiên: rank liền mạch 1…n, không bài nào xuất hiện hai lần."""
    problems = []
    by_rank = ranked.sort_values(["session_id", "rank"], kind="stable")
    expected = by_rank.groupby("session_id").cumcount() + 1
    n_bad = int((by_rank["rank"].to_numpy() != expected.to_numpy()).sum())
    if n_bad:
        problems.append(f"rank phải liền mạch 1…n trong mỗi phiên ({n_bad} dòng sai)")
    lengths = sorted(ranked.groupby("session_id").size().unique())
    if len(lengths) > 1:
        problems.append(f"mọi phiên phải có danh sách cùng độ dài, đang có {lengths}")
    n_dup = int(ranked.duplicated(["session_id", "track_id"]).sum())
    if n_dup:
        problems.append(f"track_id bị trùng trong cùng một phiên ({n_dup} dòng)")
    return problems


def scored_targets(ranked: pd.DataFrame, eval_targets: pd.DataFrame, sample: bool = False) -> pd.DataFrame:
    """Các dòng eval_targets được chấm: mọi phiên thuộc split của danh sách (sample=True: chỉ các phiên có trong danh sách).

    Split suy ra từ các phiên có trong danh sách và phải là một split duy nhất. Báo lỗi nếu danh sách
    có phiên lạ, thiếu phiên nào của split đó (trừ khi sample), hoặc chứa bài trong seed của chính phiên đó.
    """
    if ranked.empty:
        raise SchemaError("candidates", ["danh sách rỗng, không có phiên nào để chấm"])
    split_of = eval_targets.set_index("session_id")["split"]
    sessions =pd.Index(ranked["session_id"].unique())
    unknown = sessions.difference(split_of.index)
    if len(unknown):
        raise SchemaError("candidates", [f"{len(unknown)} phiên không có trong eval_targets"])
    splits = sorted(split_of[sessions].unique())
    if len(splits) != 1:
        raise SchemaError("candidates", [f"mọi phiên phải thuộc một split, đang có {splits}"])
    targets = eval_targets[eval_targets["split"] == splits[0]]
    if sample:
        targets = targets[targets["session_id"].isin(sessions)]

    problems = _list_problems(ranked)
    n_without = len(set(targets["session_id"]) - set(sessions))
    if n_without:
        problems.append(f"{n_without} phiên trong eval_targets của {splits[0]} không có ứng viên")
    seed_of = targets.set_index("session_id")["seed_track_ids"].map(set)
    in_seed = [t in s for t, s in zip(ranked["track_id"], ranked["session_id"].map(seed_of))]
    if any(in_seed):
        problems.append(f"có {sum(in_seed)} ứng viên nằm trong seed của phiên")
    if problems:
        raise SchemaError("candidates", problems)
    return targets


def evaluate(candidates: pd.DataFrame, eval_targets: pd.DataFrame, k: int = K, sample: bool = False) -> Accuracy:
    """HitRate@k, NDCG@k và MRR của một danh sách xếp hạng (cột session_id, rank, track_id).

    Nhận file candidates của một split, hoặc danh sách đã xếp lại sau re-rank. eval_targets có thể chứa cả val lẫn test.
    Mặc định mọi phiên của split phải có danh sách; sample=True để chấm một mẫu phiên (grid search #25).
    """
    targets = scored_targets(candidates, eval_targets, sample=sample)
    ranks = target_ranks(candidates, targets).to_numpy()
    return Accuracy(
        split=str(targets["split"].iloc[0]),
        n_sessions=len(ranks),
        k=k,
        list_length=int(candidates.groupby("session_id").size().iloc[0]),
        hit_rate=hit_rate(ranks, k),
        ndcg=ndcg(ranks, k),
        mrr=mrr(ranks),
    )


def main(argv: list[str] | None = None) -> Accuracy:
    ap = argparse.ArgumentParser(description="Chấm HitRate@k, NDCG@k, MRR cho một file candidates (#8)")
    ap.add_argument("--data", required=True, help="Thư mục có eval_targets.parquet (mixnext-splited hoặc data/fake)")
    ap.add_argument("--candidates", required=True, help="File candidates/{model}_{split}_run{r}.parquet")
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args(argv)
    candidates = pd.read_parquet(args.candidates)
    validate_candidates(candidates)
    acc = evaluate(candidates, pd.read_parquet(Path(args.data) / "eval_targets.parquet"), k=args.k)
    print(json.dumps(asdict(acc), ensure_ascii=False, indent=2))
    return acc


if __name__ == "__main__":
    main()
