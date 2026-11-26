"""Huấn luyện SASRec, xuất candidates val/test và checkpoint (#16).

    python -m src.models.sasrec --data <thư mục mixnext-splited hoặc data/fake> --out <thư mục ra> [--run 0]

Ghi candidates/sasrec_{val,test}_run{r}.parquet, sasrec/sasrec_run{r}.pt (checkpoint) và sasrec/sasrec_run{r}.json
(tham số, lịch sử từng epoch). Tham số cố định (SASRecParams), dừng sớm theo NDCG@10 trên một mẫu phiên val.
random_seed = số lần chạy, cho 3 lần chạy ở #26. In ra JSON: tham số, lịch sử, độ chính xác trên toàn bộ val.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.data.schema import EVAL_SPLITS, validate_dataset
from src.eval.run import VAL, evaluate
from src.models.base import export_candidates, save_candidates
from src.models.sasrec.model import SASRec, SASRecParams


def run(data_dir: str | Path, out_dir: str | Path, run_number: int = 0, params: SASRecParams = SASRecParams()) -> dict:
    data = Path(data_dir)
    tracks = pd.read_parquet(data / "tracks.parquet")
    sessions = pd.read_parquet(data / "sessions.parquet")
    eval_targets = pd.read_parquet(data / "eval_targets.parquet")

    model = SASRec(params, random_seed=run_number).fit(sessions, eval_targets)
    cands = {split: export_candidates(model, eval_targets, split) for split in EVAL_SPLITS}
    validate_dataset(tracks, sessions, eval_targets, candidates=cands)
    save_candidates(cands, out_dir, model.name, run_number)
    model.save(Path(out_dir) / model.name / f"{model.name}_run{run_number}.pt")

    report = {
        "params": asdict(params),
        "run": run_number,
        "device": str(model.device),
        "history": model.history,
        "val": asdict(evaluate(cands[VAL], eval_targets)),
    }
    Path(out_dir, model.name, f"{model.name}_run{run_number}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="SASRec: huấn luyện, xuất candidates và checkpoint (#16)")
    ap.add_argument("--data", required=True, help="Thư mục có tracks, sessions, eval_targets (.parquet)")
    ap.add_argument("--out", required=True, help="Thư mục ra")
    ap.add_argument("--run", type=int, default=0, help="Số lần chạy, cũng là random_seed (0, 1, 2)")
    ap.add_argument("--max-epochs", type=int, help="Ghi đè SASRecParams.max_epochs")
    ap.add_argument("--minutes", type=float, help="Ghi đè SASRecParams.minutes (giới hạn thời gian huấn luyện)")
    args = ap.parse_args(argv)
    overrides = {"max_epochs": args.max_epochs, "minutes": args.minutes}
    params = dataclasses.replace(SASRecParams(), **{key: value for key, value in overrides.items() if value is not None})
    report = run(args.data, args.out, args.run, params)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
