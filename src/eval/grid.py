"""Grid search (w, α) trên val: chọn trọng số điểm mix theo luật và α cho bảng kết quả chính (ticket #25, spec §6).

    python -m src.eval.grid --input /kaggle/input --out /kaggle/working/out      # ghi grid.csv và mix_params.json

Lưới: 21 bộ w (bước 0,2, tổng bằng 1) × 11 α cho điểm mix theo luật; 11 α cho điểm mix học nếu có mix_learned.npz.
Mô hình gu là mô hình app dùng (mặc định ALS: vector bài và decay đã chọn ở #14), nên cấu hình chọn ra khớp với demo.
Độ chính xác và độ mượt đo trên cùng một mẫu phiên val (val_sample, như đo độ mượt ở #19):
- độ chính xác: ứng viên của seed chấm một lần; mỗi cấu hình chỉ còn phép tính mảng (SeedCandidates);
- độ mượt: chuỗi tham lam 10 bài; các cấu hình cùng α sinh chung (generate_chains_many) để dùng chung ứng viên gu.
Chọn theo spec §6 (choose): tỉ lệ cặp mix được cao nhất mà NDCG@10 giảm ≤ 10% so với α = 1.
mix_params.json ghi theo định dạng artifacts/ (MixParams); notebook 14 Add Input thư mục này để app dùng đúng cấu hình.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import N_CANDIDATES
from src.eval.accuracy import MISS, hit_rate, mrr, ndcg
from src.eval.run import K, Accuracy
from src.eval.smoothness import smoothness
from src.eval.sweep import ALPHAS, MAX_NDCG_DROP, N_SMOOTH, RANDOM_SEED, best_row, val_sample
from src.export_artifacts import MIX_LEARNED, MIX_PARAMS, MixParams, _find, _find_optional
from src.models.base import TasteModel, candidates_for_sessions
from src.models.popularity import Popularity
from src.models.session import SessionVectorTaste
from src.rerank.combine import MixScorer, gu_norm
from src.rerank.generate import generate_chains_many
from src.rerank.mix_learned import LearnedMix, MixEncoders
from src.rerank.mix_rule import MixComponents, MixWeights, RuleMix

WEIGHT_STEPS = 5  # bước 0,2: mỗi trọng số là i/5


def weight_grid(steps: int = WEIGHT_STEPS) -> list[MixWeights]:
    """Mọi bộ (w_bpm, w_key, w_energy) là bội của 1/steps, không âm, tổng bằng 1 (steps = 5: 21 bộ)."""
    return [
        MixWeights(round(b / steps, 10), round(k / steps, 10), round((steps - b - k) / steps, 10))
        for b in range(steps, -1, -1)
        for k in range(steps - b, -1, -1)
    ]


@dataclass(frozen=True)
class SeedCandidates:
    """Ứng viên gu cho seed của một mẫu phiên, chấm một lần rồi dùng cho mọi (w, α).

    Độ chính xác của một cấu hình chỉ còn là phép tính trên mảng (số phiên × số ứng viên), cho đúng kết quả của
    rerank_candidates rồi evaluate (#19), kể cả cách xếp khi hòa điểm cuối (giữ thứ tự ứng viên).
    """

    targets: pd.DataFrame         # các dòng eval_targets của mẫu, theo thứ tự phiên
    track_ids: list[list[str]]    # ứng viên của từng phiên, theo hạng điểm gu
    gu_norm: np.ndarray           # (số phiên, số ứng viên)
    target_pos: np.ndarray        # vị trí bài đích trong ứng viên, −1 nếu không có

    @classmethod
    def build(cls, model: TasteModel, targets: pd.DataFrame, k: int = N_CANDIDATES) -> SeedCandidates:
        results = candidates_for_sessions(model, [list(seed) for seed in targets["seed_track_ids"]], k)
        track_ids = [list(ids) for ids, _ in results]
        position = [{t: i for i, t in enumerate(ids)} for ids in track_ids]
        target_pos = np.array([p.get(t, -1) for p, t in zip(position, targets["target_track_id"])], dtype=int)
        norms = np.stack([gu_norm(taste) for _, taste in results])
        return cls(targets.reset_index(drop=True), track_ids, norms, target_pos)

    @property
    def current(self) -> list[str]:
        """Bài đang phát của mỗi phiên: bài cuối seed."""
        return [seed[-1] for seed in self.targets["seed_track_ids"]]

    def rule_components(self, rule: RuleMix) -> MixComponents:
        """Ba điểm thành phần của luật cho mọi ứng viên, dạng mảng (số phiên × số ứng viên); .weighted(w) ra điểm mix."""
        parts = [rule.components(c, ids) for c, ids in zip(self.current, self.track_ids)]
        return MixComponents(*(np.stack([getattr(p, name) for p in parts]) for name in ("bpm", "key", "energy")))

    def mix_scores(self, mix: MixScorer) -> np.ndarray:
        """Điểm mix bất kỳ (vd. điểm mix học) cho mọi ứng viên, (số phiên × số ứng viên)."""
        return np.stack([np.asarray(mix(c, ids), dtype=float) for c, ids in zip(self.current, self.track_ids)])

    def accuracy(self, mix_scores: np.ndarray, alpha: float, k: int = K) -> Accuracy:
        """HitRate@k, NDCG@k, MRR sau khi xếp lại cả danh sách theo điểm cuối = α · gu_norm + (1 − α) · điểm mix."""
        if mix_scores.shape != self.gu_norm.shape:
            raise ValueError(f"điểm mix phải có shape {self.gu_norm.shape}, đang có {mix_scores.shape}")
        final = alpha * self.gu_norm + (1 - alpha) * mix_scores
        rows = np.arange(len(final))
        found = self.target_pos >= 0
        target = final[rows, np.where(found, self.target_pos, 0)][:, None]
        before = np.arange(final.shape[1])[None, :] < self.target_pos[:, None]
        ranks = 1 + (final > target).sum(axis=1) + ((final == target) & before).sum(axis=1)  # hòa: đứng sau ứng viên trước nó
        ranks = np.where(found, ranks, MISS)
        return Accuracy(
            split=str(self.targets["split"].iloc[0]), n_sessions=len(ranks), k=k, list_length=final.shape[1],
            hit_rate=hit_rate(ranks, k), ndcg=ndcg(ranks, k), mrr=mrr(ranks),
        )


RULE, LEARNED = "rule", "learned"  # cột mix của grid.csv


@dataclass(frozen=True)
class Choice:
    """Cấu hình chọn từ lưới: (w, α) của mix theo luật, α của mix học (None nếu không chạy mix học)."""

    rule_weights: MixWeights
    rule_alpha: float
    learned_alpha: float | None
    rows: list  # nhãn các dòng được chọn trong bảng (để đánh dấu cột chosen)
    warnings: list[str]


def choose(table: pd.DataFrame) -> Choice:
    """Mỗi loại mix một cấu hình theo best_row (spec §6): luật chọn cả w và α, mix học chỉ chọn α.

    Ngưỡng NDCG tính theo α = 1 của chính loại mix đó. Chỉ α = 1 đạt ngưỡng nghĩa là xếp lại không giúp được gì mà
    không mất quá 10% NDCG: ghi vào warnings để báo cả nhóm trước khi chỉnh (issue #25).
    """
    rows, warnings, alphas = [], [], {}
    for mix in (RULE, LEARNED):
        part = table[table["mix"] == mix]
        if part.empty:
            continue
        label = best_row(part)
        rows.append(label)
        alphas[mix] = float(table.loc[label, "alpha"])
        if np.isclose(alphas[mix], 1.0):
            warnings.append(f"{mix}: chỉ α = 1 giữ được NDCG@10 trong mức giảm {MAX_NDCG_DROP:.0%}; báo cả nhóm trước khi chỉnh")
    if RULE not in alphas:
        raise ValueError("bảng lưới phải có các dòng mix theo luật (mix = 'rule')")
    rule = table.loc[rows[0]]
    weights = MixWeights(float(rule["w_bpm"]), float(rule["w_key"]), float(rule["w_energy"]))
    return Choice(weights, alphas[RULE], alphas.get(LEARNED), rows, warnings)


def grid_search(
    model: TasteModel,
    tracks: pd.DataFrame,
    eval_targets: pd.DataFrame,
    learned: LearnedMix | None = None,
    weights: list[MixWeights] | None = None,
    alphas: tuple[float, ...] = ALPHAS,
    n_sessions: int = N_SMOOTH,
    random_seed: int = RANDOM_SEED,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Mỗi cấu hình một dòng: mix, w_bpm, w_key, w_energy (trống với mix học), alpha, độ chính xác, độ mượt.

    Thứ tự dòng: luật theo thứ tự weight_grid rồi α, sau đó mix học theo α (choose hòa thì lấy dòng đứng trước).
    """
    weights = weight_grid() if weights is None else weights
    sample = val_sample(eval_targets, n_sessions, random_seed)
    seeds = [list(seed) for seed in sample["seed_track_ids"]]
    rule = RuleMix(tracks)
    cands = SeedCandidates.build(model, sample)
    components = cands.rule_components(rule)
    configs: list[tuple[str, MixWeights | None, np.ndarray]] = [(RULE, w, components.weighted(w)) for w in weights]
    if learned is not None:
        configs.append((LEARNED, None, cands.mix_scores(learned)))

    def score_all(current: str, candidate_ids: Sequence[str]) -> list[np.ndarray]:
        """Điểm mix của mọi cấu hình từ một lần tra đặc trưng: ba điểm thành phần, rồi trọng số của từng bộ w."""
        parts = rule.components(current, candidate_ids)
        return [parts.weighted(w) for w in weights] + ([learned(current, candidate_ids)] if learned is not None else [])

    rows: dict[tuple[int, float], dict] = {}
    for alpha in alphas:
        t0 = time.perf_counter()
        chains = generate_chains_many(model, seeds, score_all, len(configs), alpha)
        for i, ((name, w, scores), chain) in enumerate(zip(configs, chains)):
            acc = cands.accuracy(scores, alpha)
            rows[(i, alpha)] = {
                "mix": name,
                "w_bpm": None if w is None else w.bpm,
                "w_key": None if w is None else w.key,
                "w_energy": None if w is None else w.energy,
                "alpha": alpha, "hit_rate": acc.hit_rate, "ndcg": acc.ndcg, "mrr": acc.mrr, "n_sessions": acc.n_sessions,
                **asdict(smoothness(chain, rule.features)),
            }
        log(f"α = {alpha:.1f}: {len(configs)} cấu hình, {time.perf_counter() - t0:.0f} giây")
    return pd.DataFrame([rows[(i, alpha)] for i in range(len(configs)) for alpha in alphas])


def taste_model(root: Path, model: str, tracks: pd.DataFrame) -> SessionVectorTaste:
    """Mô hình gu như app dựng từ artifacts/ (Artifacts.taste_model): vector bài, decay đã chọn, dự phòng Popularity."""
    params = json.loads(_find(root, f"{model}_params.json").read_text(encoding="utf-8"))
    vectors = np.load(_find(root, f"{model}.npy"))
    index = pd.read_parquet(_find(root, f"{model}_tracks.parquet")).sort_values("row")
    return SessionVectorTaste(model, vectors, index["track_id"].tolist(), float(params["params"]["decay"]),
                              Popularity.from_tracks(tracks))


def run_grid(input_dir: str | Path, out_dir: str | Path, model: str = "als", n_sessions: int = N_SMOOTH,
             random_seed: int = RANDOM_SEED, log: Callable[[str], None] = print) -> tuple[pd.DataFrame, Choice]:
    """Đọc dữ liệu dưới input_dir (tìm theo tên file như export_artifacts), chạy lưới, ghi grid.csv và mix_params.json."""
    root = Path(input_dir)
    tracks = pd.read_parquet(_find(root, "tracks.parquet"))
    eval_targets = pd.read_parquet(_find(root, "eval_targets.parquet"))
    learned_path = _find_optional(root, MIX_LEARNED)
    learned = None if learned_path is None else LearnedMix(tracks, MixEncoders.load(learned_path))
    table = grid_search(taste_model(root, model, tracks), tracks, eval_targets, learned, n_sessions=n_sessions,
                        random_seed=random_seed, log=log)
    choice = choose(table)
    table["chosen"] = table.index.isin(choice.rows)
    params = MixParams(
        choice.rule_weights, choice.rule_alpha, choice.learned_alpha,
        source=f"grid search #25: {model}, {int(table['n_sessions'].iloc[0])} phiên val, random_seed {random_seed} (grid.csv)",
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "grid.csv", index=False)
    (out / MIX_PARAMS).write_text(json.dumps(params.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    return table, choice


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description="Grid search (w, α) trên val, ghi grid.csv và mix_params.json (#25)")
    ap.add_argument("--input", required=True, help="Thư mục gốc chứa tracks, eval_targets, {model}.npy, {model}_tracks.parquet, "
                    "{model}_params.json và (nếu có) mix_learned.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="als", help="Mô hình gu có embedding (mặc định als, mô hình app dùng)")
    ap.add_argument("--n-sessions", type=int, default=N_SMOOTH, help="Số phiên val của mẫu")
    ap.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    table, choice = run_grid(args.input, args.out, args.model, args.n_sessions, args.random_seed)
    print(table[table["chosen"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    w = choice.rule_weights
    learned = "không có mix học" if choice.learned_alpha is None else f"mix học α = {choice.learned_alpha:.1f}"
    print(f"Chọn: luật w = {w.bpm:g}/{w.key:g}/{w.energy:g}, α = {choice.rule_alpha:.1f}; {learned} · "
          f"{time.perf_counter() - t0:.0f} giây")
    for warning in choice.warnings:
        print(f"CẢNH BÁO: {warning}")
    return table


if __name__ == "__main__":
    main()
