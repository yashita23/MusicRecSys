"""Gói những gì app cần để chạy không mạng, không torch vào artifacts/ (ticket #20, spec §8–§9).

    python -m src.export_artifacts --input /kaggle/input --out /kaggle/working/artifacts
    python -m src.export_artifacts --input data --out artifacts                       # ở máy

--input là thư mục gốc; script tìm theo tên file bên dưới (như notebook, bố cục data-contract): tracks.parquet,
eval_targets.parquet, {als,item2vec}_params.json, item_embeddings/{model}.npy và {model}_tracks.parquet,
và nếu có: mix_learned.npz (#23), mix_params.json (#25). Mô hình gu được chọn là mô hình có NDCG@10 val cao nhất
trong params.json.

artifacts/
├── tracks.parquet           # schema tracks (có tên bài, nghệ sĩ): bài có vector ∪ bài của playlist demo
├── item_embeddings.npy      # vector bài float32 (đã chuẩn hóa nếu mô hình chọn normalize)
├── track_index.parquet      # row, track_id
├── demo_playlists.parquet   # session_id, seed_track_ids, target_track_id: phiên test mẫu cho app chọn
├── mix_params.json          # trọng số và α của mix theo luật, α của mix học (mặc định, hoặc #25 chọn)
├── mix_learned.npz          # điểm mix học (#23), nếu có
└── manifest.json            # mô hình, decay, NDCG val của từng mô hình, số bài, số bài bị cắt, dung lượng

load_artifacts chỉ dùng numpy và pandas: app dựng mô hình gu (vector phiên, dự phòng Popularity theo
n_occurrences), điểm mix theo luật và điểm mix học từ đây.
"""
from __future__ import annotations

import argparse
import json
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
MAX_BYTES = 200_000_000        # tiêu chí #20: tổng dung lượng artifacts < 200 MB
TRACK_ROW_BYTES = 120          # ước lượng một dòng tracks.parquet (tên bài, nghệ sĩ, đặc trưng) để cắt danh mục trước khi ghi
N_DEMO = 100
RANDOM_SEED = 42
TEST = "test"
DEFAULT_ALPHA = 0.7            # α được chọn ở #19 (reports/sweep_v1.csv) cho mix theo luật với trọng số mặc định

TRACKS, VECTORS, INDEX, DEMO = "tracks.parquet", "item_embeddings.npy", "track_index.parquet", "demo_playlists.parquet"
MIX_PARAMS, MIX_LEARNED, MANIFEST = "mix_params.json", "mix_learned.npz", "manifest.json"


@dataclass(frozen=True)
class MixParams:
    """Cấu hình mix app dùng (mix_params.json): trọng số và α của mix theo luật, α của mix học.

    Grid search #25 ghi lại file này theo cùng định dạng; khi chưa có thì dùng trọng số mặc định (#12) và α của #19.
    """

    rule_weights: MixWeights = MixWeights()
    rule_alpha: float = DEFAULT_ALPHA
    learned_alpha: float | None = None  # #25 chọn; chưa có thì app dùng rule_alpha
    source: str = "mặc định: trọng số của #12, α của #19 (reports/sweep_v1.csv)"

    def __post_init__(self) -> None:
        for name, alpha in (("rule.alpha", self.rule_alpha), ("learned.alpha", self.learned_alpha)):
            if alpha is not None and not 0.0 <= alpha <= 1.0:
                raise ValueError(f"mix_params: {name} phải trong khoảng 0–1, đang là {alpha}")

    def to_json(self) -> dict[str, Any]:
        return {"rule": {"weights": asdict(self.rule_weights), "alpha": self.rule_alpha},
                "learned": {"alpha": self.learned_alpha}, "source": self.source}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> MixParams:
        try:
            return cls(MixWeights(**data["rule"]["weights"]), float(data["rule"]["alpha"]),
                       None if data["learned"]["alpha"] is None else float(data["learned"]["alpha"]), str(data["source"]))
        except (KeyError, TypeError) as e:
            raise ValueError(f"mix_params sai định dạng (xem docs/data-contract.md, mục artifacts/): {e!r}") from e

    @classmethod
    def load(cls, path: str | Path) -> MixParams:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Artifacts:
    """Nội dung của artifacts/: đủ để app gợi ý mà không cần sessions, implicit, gensim hay torch."""

    tracks: pd.DataFrame
    vectors: np.ndarray            # hàng i ứng với track_index.row == i
    track_index: pd.DataFrame      # row, track_id
    demo: pd.DataFrame             # session_id, seed_track_ids, target_track_id
    model: str
    decay: float
    mix_params: MixParams = MixParams()
    mix_learned: MixEncoders | None = None
    info: dict[str, Any] = field(default_factory=dict)  # chỉ để đọc (manifest): NDCG val, tham số đầy đủ, số bài bị cắt

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
        return RuleMix(self.tracks, self.mix_params.rule_weights)

    def learned_mix(self) -> LearnedMix | None:
        return None if self.mix_learned is None else LearnedMix(self.tracks, self.mix_learned)


def demo_playlists(eval_targets: pd.DataFrame, n: int = N_DEMO, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """n phiên test cố định theo random_seed (ít hơn n thì lấy hết), theo session_id."""
    test = eval_targets[eval_targets["split"] == TEST].sort_values("session_id")
    if len(test) > n:
        test = test.iloc[np.sort(np.random.default_rng(random_seed).choice(len(test), n, replace=False))]
    return test[["session_id", "seed_track_ids", "target_track_id"]].reset_index(drop=True)


def demo_track_ids(demo: pd.DataFrame) -> set[str]:
    """Mọi bài của các playlist demo: seed và bài đích."""
    return {t for seed in demo["seed_track_ids"] for t in seed} | set(demo["target_track_id"])


def build_artifacts(
    tracks: pd.DataFrame,
    vectors: np.ndarray,
    track_ids: pd.Series | list[str],
    model: str,
    decay: float,
    demo: pd.DataFrame,
    mix_params: MixParams = MixParams(),
    mix_learned: MixEncoders | None = None,
    max_bytes: int = MAX_BYTES,
    info: dict[str, Any] | None = None,
) -> Artifacts:
    """Giữ mọi bài có vector; vượt max_bytes thì bỏ bài ít lượt nhất (n_occurrences), luôn giữ bài của playlist demo.

    Thứ tự hàng của vector giữ như file gốc, vì SessionVectorTaste xếp hòa điểm theo thứ tự này.
    """
    ids = pd.Index(list(track_ids))
    demo_ids = demo_track_ids(demo)
    missing = demo_ids - set(tracks["track_id"])
    if missing:
        raise ValueError(f"{len(missing)} bài của playlist demo không có trong tracks")
    plays = tracks.set_index("track_id")["n_occurrences"].reindex(ids).fillna(0).to_numpy()

    per_track = vectors.shape[1] * vectors.dtype.itemsize + 2 * TRACK_ROW_BYTES  # vector, dòng tracks, dòng track_index
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
    return Artifacts(out_tracks, np.ascontiguousarray(vectors[keep]), index, demo, model, float(decay), mix_params, mix_learned, meta)


def write_artifacts(artifacts: Artifacts, out_dir: str | Path) -> int:
    """Ghi artifacts/ (xóa mix_learned.npz cũ nếu lần này không có); trả về tổng dung lượng, cũng ghi vào manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts.tracks.to_parquet(out / TRACKS, index=False)
    np.save(out / VECTORS, artifacts.vectors)
    artifacts.track_index.to_parquet(out / INDEX, index=False)
    artifacts.demo.to_parquet(out / DEMO, index=False)
    (out / MIX_PARAMS).write_text(json.dumps(artifacts.mix_params.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    if artifacts.mix_learned is not None:
        artifacts.mix_learned.save(out / MIX_LEARNED)
    else:
        (out / MIX_LEARNED).unlink(missing_ok=True)
    data_files = [TRACKS, VECTORS, INDEX, DEMO, MIX_PARAMS] + ([MIX_LEARNED] if artifacts.mix_learned is not None else [])
    manifest = {
        "model": artifacts.model,
        "decay": artifacts.decay,
        "mix_learned": artifacts.mix_learned is not None,
        "n_tracks": len(artifacts.tracks),
        "n_tracks_in_index": len(artifacts.track_index),
        "dim": int(artifacts.vectors.shape[1]),
        "n_demo": len(artifacts.demo),
        **artifacts.info,
        "bytes": sum((out / name).stat().st_size for name in data_files),
    }
    (out / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(manifest["bytes"]) + (out / MANIFEST).stat().st_size


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
    model, decay, has_learned = manifest.pop("model"), float(manifest.pop("decay")), manifest.pop("mix_learned")
    learned = MixEncoders.load(root / MIX_LEARNED) if has_learned else None
    return Artifacts(tracks, vectors, index, demo, model, decay, MixParams.load(root / MIX_PARAMS), learned, manifest)


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
    loaded, ndcg = {}, {}
    for model, path in params_files.items():
        try:
            loaded[model] = json.loads(path.read_text(encoding="utf-8"))
            ndcg[model] = max(float(row["ndcg"]) for row in loaded[model]["grid"])
            float(loaded[model]["params"]["decay"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{path}: thiếu params.decay hoặc grid[].ndcg (xem docs/data-contract.md): {e!r}") from e
    best = max(ndcg, key=lambda m: ndcg[m])
    return best, loaded[best]["params"], ndcg


def export(input_dir: str | Path, out_dir: str | Path, n_demo: int = N_DEMO, random_seed: int = RANDOM_SEED,
           max_bytes: int = MAX_BYTES) -> Artifacts:
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
    mix_params_path, learned_path = _find_optional(root, MIX_PARAMS), _find_optional(root, MIX_LEARNED)

    artifacts = build_artifacts(
        tracks, vectors, index.sort_values("row")["track_id"], model, float(params["decay"]),
        demo_playlists(eval_targets, n_demo, random_seed),
        mix_params=MixParams.load(mix_params_path) if mix_params_path else MixParams(),
        mix_learned=MixEncoders.load(learned_path) if learned_path else None,
        max_bytes=max_bytes,
        info={"params": params, "val_ndcg": ndcg, "random_seed": random_seed},
    )
    total = write_artifacts(artifacts, out_dir)
    if total >= max_bytes:
        raise ValueError(f"artifacts nặng {total / 1e6:.1f} MB, không dưới {max_bytes / 1e6:.0f} MB: "
                         f"tăng TRACK_ROW_BYTES để cắt mạnh hơn rồi chạy lại")
    return artifacts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Xuất artifacts/ cho app (#20)")
    ap.add_argument("--input", required=True, help="Thư mục gốc chứa dữ liệu, embedding và params.json (tìm theo tên file)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-demo", type=int, default=N_DEMO, help="Số playlist test mẫu cho app chọn")
    ap.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args(argv)
    try:
        artifacts = export(args.input, args.out, args.n_demo, args.random_seed)
    except ValueError as e:
        ap.error(str(e))
    manifest = json.loads((Path(args.out) / MANIFEST).read_text(encoding="utf-8"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"{artifacts.model}: {len(artifacts.track_index):,} bài có vector, {len(artifacts.tracks):,} bài trong tracks, "
          f"{len(artifacts.demo)} playlist demo · {manifest['bytes'] / 1e6:.1f} MB · mix: {artifacts.mix_params.source}")


if __name__ == "__main__":
    main()
