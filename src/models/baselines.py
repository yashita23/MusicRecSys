"""Xuất candidates của hai baseline và chấm trên val (ticket #9).

    python -m src.models.baselines --data <thư mục mixnext-splited hoặc data/fake> --out <thư mục ra>

Ghi candidates/{popularity,itemknn}_{val,test}_run0.parquet. Hai baseline tất định nên chỉ có lần chạy 0.
In ra JSON: tên mô hình → HitRate@10, NDCG@10, MRR trên val (mốc cho ngưỡng NDCG ở spec §6).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.data.schema import EVAL_SPLITS, validate_dataset
from src.eval.run import Accuracy, evaluate
from src.models.base import TasteModel, candidates_path, export_candidates
from src.models.item_knn import ItemKNN
from src.models.popularity import Popularity


def run_baselines(data_dir: str | Path, out_dir: str | Path) -> dict[str, Accuracy]:
    data = Path(data_dir)
    tracks = pd.read_parquet(data / "tracks.parquet")
    sessions = pd.read_parquet(data / "sessions.parquet")
    eval_targets = pd.read_parquet(data / "eval_targets.parquet")
    models: list[TasteModel] = [Popularity(), ItemKNN()]

    report = {}
    for model in models:
        model.fit(sessions)
        cands = {split: export_candidates(model, eval_targets, split) for split in EVAL_SPLITS}
        validate_dataset(tracks, sessions, eval_targets, candidates=cands)
        for split, df in cands.items():
            path = candidates_path(out_dir, model.name, split)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
        report[model.name] = evaluate(cands["val"], eval_targets)
    return report


def main(argv: list[str] | None = None) -> dict[str, Accuracy]:
    ap = argparse.ArgumentParser(description="Xuất candidates của Popularity và item-kNN liền kề, chấm trên val (#9)")
    ap.add_argument("--data", required=True, help="Thư mục có tracks, sessions, eval_targets (.parquet)")
    ap.add_argument("--out", required=True, help="Thư mục ra; file nằm trong <out>/candidates/")
    args = ap.parse_args(argv)
    report = run_baselines(args.data, args.out)
    print(json.dumps({name: asdict(acc) for name, acc in report.items()}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
