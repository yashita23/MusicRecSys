"""Gói những gì app cần để chạy không mạng, không torch vào artifacts/ (ticket #20, spec §8–§9).

    python -m src.export_artifacts --input /kaggle/input --out /kaggle/working/artifacts
    python -m src.export_artifacts --input data --out artifacts                       # ở máy

--input là thư mục gốc; script tìm theo tên file bên dưới (như notebook, bố cục data-contract): tracks.parquet,
eval_targets.parquet, {als,item2vec}_params.json, item_embeddings/{model}.npy và {model}_tracks.parquet,
và mix_learned.npz nếu có. Mô hình gu được chọn là mô hình có NDCG@10 val cao nhất trong params.json.

artifacts/
├── tracks.parquet           # schema tracks (có tên bài, nghệ sĩ): bài có vector ∪ bài của playlist demo
├── item_embeddings.npy      # vector bài float32 (đã chuẩn hóa nếu mô hình chọn normalize)
├── track_index.parquet      # row, track_id
├── demo_playlists.parquet   # session_id, seed_track_ids, target_track_id: phiên test mẫu cho app chọn
├── manifest.json            # mô hình, decay, trọng số mix theo luật, NDCG val của từng mô hình, số bài, dung lượng
└── mix_learned.npz          # điểm mix học (#23), nếu có

load_artifacts chỉ dùng numpy và pandas: app dựng mô hình gu (vector phiên, dự phòng Popularity theo
n_occurrences), điểm mix theo luật và điểm mix học từ đây.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.schema import validate_eval_targets, validate_item_embeddings, validate_tracks
from src.models.popularity import Popularity
from src.models.session import SessionVectorTaste
from src.rerank.mix_learned import LearnedMix, MixEncoders
from src.rerank.mix_rule import MixWeights, RuleMix

MODELS = ("als", "item2vec")  # mô hình gu có embedding mà app tính lại được bằng numpy (spec §8 mục 5)
MAX_BYTES = 200 * 1024**2      # tiêu chí #20: tổng dung lượng artifacts < 200 MB
TRACK_ROW_BYTES = 120          # ước lượng một dòng tracks.parquet (tên bài, nghệ sĩ, đặc trưng) để cắt danh mục trước khi ghi
N_DEMO = 100
RANDOM_SEED = 42
TEST = "test"

TRACKS, VECTORS, INDEX, DEMO, MANIFEST, MIX_LEARNED = (
    "tracks.parquet", "item_embeddings.npy", "track_index.parquet", "demo_playlists.parquet", "manifest.json", "mix_learned.npz"
)


@dataclass(frozen=True)
class Artifacts:
    """Nội dung của artifacts/: đủ để app gợi ý mà không cần sessions, implicit, gensim hay torch."""

    tracks: pd.DataFrame
    vectors: np.ndarray            # hàng i ứng với track_index.row == i
    track_index: pd.DataFrame      # row, track_id
    demo: pd.DataFrame             # session_id, seed_track_ids, target_track_id
    model: str
    decay: float
    mix_weights: MixWeights = MixWeights()
    info: dict[str, Any] = field(default_factory=dict)  # chỉ để đọc: NDCG val, tham số đầy đủ, số bài bị cắt
    mix_learned: MixEncoders | None = None

    @property
    def nbytes(self) -> int:
        """Dung lượng ước lượng trước khi ghi (vector + bảng bài); dung lượng thật kiểm tra sau khi ghi."""
        return int(self.vectors.nbytes + TRACK_ROW_BYTES * (len(self.tracks) + len(self.track_index)))

    def taste_model(self) -> SessionVectorTaste:
        """Mô hình gu như lúc đánh giá: vector phiên với decay đã chọn; phiên không bài nào có vector thì dùng Popularity."""
        return SessionVectorTaste(
            self.model, self.vectors, self.track_index["track_id"].tolist(), self.decay, Popularity.from_tracks(self.tracks)
        )

    def rule_mix(self) -> RuleMix:
        return RuleMix(self.tracks, self.mix_weights)

    def learned_mix(self) -> LearnedMix | None:
        return None if self.mix_learned is None else LearnedMix(self.tracks, self.mix_learned)


def demo_playlists(eval_targets: pd.DataFrame, n: int = N_DEMO, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """n phiên test cố định theo random_seed (ít hơn n thì lấy hết), theo session_id."""
    test = eval_targets[eval_targets["split"] == TEST].sort_values("session_id")
    if len(test) > n:
        test = test.iloc[np.sort(np.random.default_rng(random_seed).choice(len(test), n, replace=False))]
    return test[["session_id", "seed_track_ids", "target_track_id"]].reset_index(drop=True)


def _demo_track_ids(demo: pd.DataFrame) -> set[str]:
    return {t for seed in demo["seed_track_ids"] for t in seed} | set(demo["target_track_id"])


def build_artifacts(
    tracks: pd.DataFrame,
    vectors: np.ndarray,
    track_ids: pd.Series | list[str],
    model: str,
    decay: float,
    demo: pd.DataFrame,
    mix_weights: MixWeights = MixWeights(),
    max_bytes: int = MAX_BYTES,
    info: dict[str, Any] | None = None,
) -> Artifacts:
    """Giữ mọi bài có vector; vượt max_bytes thì bỏ bài ít lượt nhất (n_occurrences), luôn giữ bài của playlist demo.

    Thứ tự hàng của vector giữ như file gốc, vì SessionVectorTaste xếp hòa điểm theo thứ tự này.
    """
    ids = pd.Index(list(track_ids))
    demo_ids = _demo_track_ids(demo)
    missing = demo_ids - set(tracks["track_id"])
    if missing:
        raise ValueError(f"{len(missing)} bài của playlist demo không có trong tracks")
    plays = tracks.set_index("track_id")["n_occurrences"].reindex(ids).fillna(0).to_numpy()

    per_track = vectors.shape[1] * vectors.dtype.itemsize + 2 * TRACK_ROW_BYTES
    n_extra = len(demo_ids - set(ids))  # bài demo không có vector: chỉ thêm dòng tracks
    budget = (max_bytes - n_extra * TRACK_ROW_BYTES) // per_track
    is_demo = ids.isin(demo_ids)
    if budget < is_demo.sum():
        raise ValueError(f"max_bytes = {max_bytes} không đủ cho {is_demo.sum()} bài có vector của playlist demo")
    keep = np.ones(len(ids), dtype=bool)
    if budget < len(ids):
        # demo trước, rồi nhiều lượt trước; hòa thì theo thứ tự hàng (ổn định)
        order = np.lexsort((np.arange(len(ids)), -plays, ~is_demo))
        keep[:] = False
        keep[order[:budget]] = True

    kept_ids = ids[keep]
    out_tracks = tracks[tracks["track_id"].isin(set(kept_ids) | demo_ids)].reset_index(drop=True)
    index = pd.DataFrame({"row": np.arange(len(kept_ids)), "track_id": kept_ids.to_numpy(dtype=object)})
    meta = dict(info or {}, n_tracks_with_vector=int(len(ids)), n_dropped=int((~keep).sum()))
    return Artifacts(out_tracks, np.ascontiguousarray(vectors[keep]), index, demo, model, float(decay), mix_weights, meta)


def write_artifacts(artifacts: Artifacts, out_dir: str | Path, mix_learned: str | Path | None = None) -> list[Path]:
    """Ghi artifacts/ và manifest.json; mix_learned (đường dẫn .npz của #23) được chép nguyên nếu có."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts.tracks.to_parquet(out / TRACKS, index=False)
    np.save(out / VECTORS, artifacts.vectors)
    artifacts.track_index.to_parquet(out / INDEX, index=False)
    artifacts.demo.to_parquet(out / DEMO, index=False)
    if mix_learned is not None:
        MixEncoders.load(mix_learned)  # báo lỗi ngay nếu trọng số không khớp đặc trưng hiện tại
        shutil.copyfile(mix_learned, out / MIX_LEARNED)
    manifest = {
        "model": artifacts.model,
        "decay": artifacts.decay,
        "mix_weights": asdict(artifacts.mix_weights),
        "mix_learned": mix_learned is not None,
        "n_tracks": len(artifacts.tracks),
        "n_tracks_in_index": len(artifacts.track_index),
        "dim": int(artifacts.vectors.shape[1]),
        "n_demo": len(artifacts.demo),
        **artifacts.info,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return [out / name for name in (TRACKS, VECTORS, INDEX, DEMO, MANIFEST)] + ([out / MIX_LEARNED] if mix_learned else [])


def load_artifacts(path: str | Path) -> Artifacts:
    """Đọc artifacts/ (chỉ numpy, pandas) và kiểm tra tracks, vector và index khớp nhau."""
    root = Path(path)
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    tracks = pd.read_parquet(root / TRACKS)
    vectors = np.load(root / VECTORS)
    index = pd.read_parquet(root / INDEX)
    demo = pd.read_parquet(root / DEMO)
    demo["seed_track_ids"] = demo["seed_track_ids"].map(list)
    validate_tracks(tracks)
    validate_item_embeddings(vectors, index)
    unknown = set(index["track_id"]) - set(tracks["track_id"])
    if unknown:
        raise ValueError(f"{root}: {len(unknown)} bài có vector nhưng không có trong {TRACKS}")
    known = {k: manifest.pop(k) for k in ("model", "decay", "mix_weights", "mix_learned")}
    learned = MixEncoders.load(root / MIX_LEARNED) if known["mix_learned"] else None
    return Artifacts(tracks, vectors, index, demo, known["model"], float(known["decay"]), MixWeights(**known["mix_weights"]), manifest, learned)


# --- CLI: tìm file đầu vào, chọn mô hình, ghi và kiểm tra dung lượng ---


def _find_optional(root: Path, name: str) -> Path | None:
    """File duy nhất tên name dưới root (None nếu không có); trùng tên thì báo lỗi để khỏi đọc nhầm bộ dữ liệu."""
    found = sorted(root.rglob(name))
    if len(found) > 1:
        raise ValueError(f"có {len(found)} file {name} dưới {root}: {[str(p) for p in found]}")
    return found[0] if found else None


def _find(root: Path, name: str) -> Path:
    path = _find_optional(root, name)
    if path is None:
        raise ValueError(f"không tìm thấy {name} dưới {root}")
    return path


def choose_model(params_files: dict[str, Path]) -> tuple[str, dict[str, Any], dict[str, float]]:
    """(mô hình, tham số đã chọn, NDCG@10 val tốt nhất của từng mô hình) theo {model}_params.json."""
    if not params_files:
        raise ValueError(f"không có {{model}}_params.json nào của {', '.join(MODELS)}")
    loaded = {m: json.loads(p.read_text(encoding="utf-8")) for m, p in params_files.items()}
    ndcg = {m: max(float(row["ndcg"]) for row in report["grid"]) for m, report in loaded.items()}
    best = max(ndcg, key=lambda m: ndcg[m])
    return best, loaded[best]["params"], ndcg


def export(input_dir: str | Path, out_dir: str | Path, n_demo: int = N_DEMO, random_seed: int = RANDOM_SEED,
           mix_weights: MixWeights = MixWeights(), max_bytes: int = MAX_BYTES) -> Artifacts:
    root = Path(input_dir)
    params_files = {m: p for m in MODELS if (p := _find_optional(root, f"{m}_params.json")) is not None}
    model, params, ndcg = choose_model(params_files)
    tracks = pd.read_parquet(_find(root, "tracks.parquet"))
    eval_targets = pd.read_parquet(_find(root, "eval_targets.parquet"))
    vectors = np.load(_find(root, f"{model}.npy"))
    index = pd.read_parquet(_find(root, f"{model}_tracks.parquet"))
    validate_tracks(tracks)
    validate_eval_targets(eval_targets)
    validate_item_embeddings(vectors, index)

    demo = demo_playlists(eval_targets, n_demo, random_seed)
    info = {"params": params, "val_ndcg": ndcg, "random_seed": random_seed}
    artifacts = build_artifacts(tracks, vectors, index.sort_values("row")["track_id"], model, params["decay"], demo,
                                mix_weights, max_bytes, info)
    written = write_artifacts(artifacts, out_dir, _find_optional(root, MIX_LEARNED))
    total = sum(p.stat().st_size for p in written)
    if total >= max_bytes:
        raise ValueError(f"artifacts nặng {total / 1024**2:.1f} MB, vượt giới hạn {max_bytes / 1024**2:.0f} MB")
    return artifacts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Xuất artifacts/ cho app (#20)")
    ap.add_argument("--input", required=True, help="Thư mục gốc chứa dữ liệu, embedding và params.json (tìm theo tên file)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-demo", type=int, default=N_DEMO, help="Số playlist test mẫu cho app chọn")
    ap.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--mix-weights", default="0.4/0.4/0.2", help="Trọng số bpm/key/energy của điểm mix theo luật (#25 chọn lại)")
    args = ap.parse_args(argv)
    try:
        weights = MixWeights(*(float(w) for w in args.mix_weights.split("/")))
        artifacts = export(args.input, args.out, args.n_demo, args.random_seed, weights)
    except (ValueError, TypeError) as e:
        ap.error(str(e))
    out = Path(args.out)
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(json.dumps(json.loads((out / MANIFEST).read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    print(f"{artifacts.model}: {len(artifacts.track_index):,} bài có vector, {len(artifacts.tracks):,} bài trong tracks, "
          f"{len(artifacts.demo)} playlist demo · {total / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
