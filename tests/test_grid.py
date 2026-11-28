"""Grid search (w, α) trên val (ticket #25, spec §6)."""
import pandas as pd
import pytest

from src.data.fake import FAKE_MODEL, make_fake_dataset
from src.eval.grid import SeedCandidates, weight_grid
from src.eval.run import evaluate
from src.eval.sweep import smoothness_seeds, val_sample
from src.models.base import export_candidates
from src.models.popularity import Popularity
from src.models.session import SessionVectorTaste
from src.rerank.combine import rerank_candidates
from src.rerank.mix_rule import MixWeights, RuleMix


@pytest.fixture(scope="module")
def fake():
    return make_fake_dataset(n_sessions=200, n_tracks=400, random_seed=4)


@pytest.fixture(scope="module")
def model(fake):
    vectors, index = fake.embeddings[FAKE_MODEL]
    return SessionVectorTaste(FAKE_MODEL, vectors, index["track_id"].tolist(), 0.7, Popularity().fit(fake.sessions))


def test_weight_grid_has_21_weight_triples_in_steps_of_0_2_summing_to_1():
    grid = weight_grid()
    assert len(grid) == 21  # số nghiệm của a + b + c = 5 với a, b, c ≥ 0: C(7, 2) = 21
    assert len(set(grid)) == 21
    for w in grid:
        assert round(w.bpm + w.key + w.energy, 9) == 1.0
        assert all(round(x * 5, 9) == round(x * 5) for x in (w.bpm, w.key, w.energy))
    assert MixWeights(0.4, 0.4, 0.2) in grid  # trọng số mặc định của #12 nằm trong lưới
    assert MixWeights(1.0, 0.0, 0.0) in grid and MixWeights(0.0, 0.0, 1.0) in grid


@pytest.mark.parametrize("weights", [MixWeights(0.4, 0.4, 0.2), MixWeights(0.0, 1.0, 0.0)])  # chỉ tông: nhiều điểm hòa
@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.7, 1.0])
def test_vectorized_accuracy_matches_reranking_each_session(fake, model, weights, alpha):
    # Mẫu phiên val như grid search; chuẩn là đường đi của #19: export_candidates → rerank_candidates → evaluate.
    sample = fake.eval_targets[fake.eval_targets["split"] == "val"].iloc[::2]
    mix = RuleMix(fake.tracks, weights)
    expected = evaluate(rerank_candidates(export_candidates(model, sample, "val"), sample, mix, alpha, k=None), sample)
    cands = SeedCandidates.build(model, sample)
    assert cands.accuracy(cands.rule_components(RuleMix(fake.tracks)).weighted(weights), alpha) == expected


def test_accuracy_and_smoothness_use_the_same_val_sample(fake):
    sample = val_sample(fake.eval_targets, n=15, random_seed=0)
    assert [list(s) for s in sample["seed_track_ids"]] == smoothness_seeds(fake.eval_targets, 15, 0)


def grid_row(mix, w, alpha, ndcg, mixable):
    weights = (None, None, None) if w is None else w
    return {"mix": mix, "w_bpm": weights[0], "w_key": weights[1], "w_energy": weights[2], "alpha": alpha,
            "ndcg": ndcg, "mixable_pct": mixable}


def test_choose_takes_most_mixable_weights_and_alpha_within_10_percent_ndcg_drop_and_only_alpha_for_learned_mix():
    from src.eval.grid import choose

    # Luật: NDCG ở α = 1 là 0,10 → ngưỡng 0,09. (0,2/0,2/0,6, α 0,5) mượt nhất nhưng NDCG 0,08 < ngưỡng, nên chọn
    # (0,6/0,2/0,2, α 0,5): 30% cặp mix được. Mix học: ngưỡng 0,09 theo NDCG của chính nó ở α = 1, chọn α 0,4.
    table = pd.DataFrame([
        grid_row("rule", (0.6, 0.2, 0.2), 1.0, 0.10, 5.0),
        grid_row("rule", (0.6, 0.2, 0.2), 0.5, 0.095, 30.0),
        grid_row("rule", (0.6, 0.2, 0.2), 0.0, 0.02, 90.0),
        grid_row("rule", (0.2, 0.2, 0.6), 1.0, 0.10, 5.0),
        grid_row("rule", (0.2, 0.2, 0.6), 0.5, 0.08, 40.0),
        grid_row("rule", (0.2, 0.2, 0.6), 0.0, 0.02, 95.0),
        grid_row("learned", None, 1.0, 0.10, 5.0),
        grid_row("learned", None, 0.4, 0.091, 12.0),
        grid_row("learned", None, 0.2, 0.05, 20.0),
    ])
    chosen = choose(table)
    assert chosen.rule_weights == MixWeights(0.6, 0.2, 0.2)
    assert chosen.rule_alpha == 0.5
    assert chosen.learned_alpha == 0.4
    assert chosen.warnings == []


def test_ties_keep_the_larger_alpha_then_the_earlier_weights_in_the_grid():
    from src.eval.grid import choose

    table = pd.DataFrame([
        grid_row("rule", (1.0, 0.0, 0.0), 1.0, 0.10, 5.0),
        grid_row("rule", (1.0, 0.0, 0.0), 0.8, 0.10, 20.0),
        grid_row("rule", (1.0, 0.0, 0.0), 0.6, 0.10, 20.0),
        grid_row("rule", (0.0, 1.0, 0.0), 1.0, 0.10, 5.0),
        grid_row("rule", (0.0, 1.0, 0.0), 0.8, 0.10, 20.0),
    ])
    chosen = choose(table)
    assert (chosen.rule_weights, chosen.rule_alpha) == (MixWeights(1.0, 0.0, 0.0), 0.8)
    assert chosen.learned_alpha is None  # không có dòng mix học


def test_warns_when_only_alpha_one_meets_the_threshold():
    from src.eval.grid import choose

    # Issue #25: không có (w, α) nào thỏa ngưỡng (ngoài α = 1, không đổi gì) thì phải báo cả nhóm trước khi chỉnh.
    table = pd.DataFrame([
        grid_row("rule", (0.4, 0.4, 0.2), 1.0, 0.10, 5.0),
        grid_row("rule", (0.4, 0.4, 0.2), 0.9, 0.05, 30.0),
    ])
    chosen = choose(table)
    assert chosen.rule_alpha == 1.0
    assert len(chosen.warnings) == 1 and "rule" in chosen.warnings[0]


def test_when_only_alpha_one_passes_the_default_weights_are_kept_instead_of_the_first_grid_entry():
    from src.eval.grid import choose

    # Ở α = 1 mọi bộ w hòa nhau (w không ảnh hưởng điểm cuối); không được đổi app sang bộ đầu lưới (1/0/0, chỉ BPM).
    table = pd.DataFrame([
        grid_row("rule", (1.0, 0.0, 0.0), 1.0, 0.10, 5.0),
        grid_row("rule", (1.0, 0.0, 0.0), 0.5, 0.05, 30.0),
        grid_row("rule", (0.4, 0.4, 0.2), 1.0, 0.10, 5.0),
        grid_row("rule", (0.4, 0.4, 0.2), 0.5, 0.05, 40.0),
    ])
    chosen = choose(table)
    assert (chosen.rule_weights, chosen.rule_alpha) == (MixWeights(0.4, 0.4, 0.2), 1.0)
    assert chosen.warnings


def write_grid_inputs(fake, root):
    """Bố cục như Kaggle Input: dữ liệu đã chia, embedding và params của ALS (dùng vector giả), mix_learned.npz."""
    import json

    import numpy as np

    from src.data.fake import write_fake
    from src.rerank.mix_learned import FEATURE_NAMES, MixEncoders

    write_fake(fake, root / "splited")
    vectors, index = fake.embeddings[FAKE_MODEL]
    emb = root / "embeddings" / "item_embeddings"
    emb.mkdir(parents=True)
    np.save(emb / "als.npy", vectors)
    index.to_parquet(emb / "als_tracks.parquet", index=False)
    (root / "embeddings" / "als_params.json").write_text(
        json.dumps({"params": {"decay": 0.7}, "grid": [{"decay": 0.7, "ndcg": 0.05}]}), encoding="utf-8")
    w = np.random.default_rng(0).normal(size=(4, len(FEATURE_NAMES))).astype(np.float32)
    MixEncoders([(w, np.zeros(4, np.float32))], [(w[::-1].copy(), np.zeros(4, np.float32))], 0.1).save(
        root / "learned" / "mix_learned.npz")


def test_cli_writes_grid_table_and_mix_params_that_artifacts_pick_up(fake, tmp_path):
    from src.eval.grid import main
    from src.export_artifacts import MixParams, export

    write_grid_inputs(fake, tmp_path / "input")
    table = main(["--input", str(tmp_path / "input"), "--out", str(tmp_path / "input" / "grid"), "--n-sessions", "12"])
    saved = pd.read_csv(tmp_path / "input" / "grid" / "grid.csv")
    assert len(saved) == 21 * 11 + 11  # 21 bộ w × 11 α cho luật, 11 α cho mix học
    assert saved.groupby("mix").size().to_dict() == {"learned": 11, "rule": 231}
    assert (saved["n_sessions"] == 12).all() and (saved["n_pairs"] == 12 * 10).all()
    assert saved["chosen"].sum() == 2  # một dòng luật, một dòng mix học
    assert len(table) == len(saved)

    params = MixParams.load(tmp_path / "input" / "grid" / "mix_params.json")
    rule = saved[saved["chosen"] & (saved["mix"] == "rule")].iloc[0]
    learned = saved[saved["chosen"] & (saved["mix"] == "learned")].iloc[0]
    assert params.rule_weights == MixWeights(rule["w_bpm"], rule["w_key"], rule["w_energy"])
    assert (params.rule_alpha, params.learned_alpha) == (rule["alpha"], learned["alpha"])
    assert "#25" in params.source and "als" in params.source  # mô hình chọn như export_artifacts: chỉ có als_params.json

    artifacts = export(tmp_path / "input", tmp_path / "artifacts", n_demo=5)  # notebook 14 Add Input cả output grid
    assert artifacts.mix_params == params


def test_grid_smoothness_matches_generating_each_configuration_alone(fake, model):
    # Chuẩn là sinh chuỗi riêng từng cấu hình như #19 (generate_chains → smoothness).
    from dataclasses import asdict

    from src.eval.grid import grid_search
    from src.eval.smoothness import smoothness
    from src.rerank.generate import generate_chains
    from src.rerank.mix_learned import FEATURE_NAMES, LearnedMix, MixEncoders

    import numpy as np

    w = np.random.default_rng(1).normal(size=(4, len(FEATURE_NAMES))).astype(np.float32)
    learned = LearnedMix(fake.tracks, MixEncoders([(w, np.zeros(4, np.float32))], [(w, np.zeros(4, np.float32))], 0.1))
    weights = [MixWeights(0.0, 1.0, 0.0), MixWeights(0.4, 0.4, 0.2)]
    table = grid_search(model, fake.tracks, fake.eval_targets, learned, weights=weights, alphas=(0.2, 0.6), n_sessions=12,
                        random_seed=3, log=lambda _: None)
    seeds = smoothness_seeds(fake.eval_targets, 12, 3)
    rule = RuleMix(fake.tracks)
    for _, row in table.iterrows():
        mix = learned if row["mix"] == "learned" else rule.with_weights(MixWeights(row["w_bpm"], row["w_key"], row["w_energy"]))
        expected = asdict(smoothness(generate_chains(model, seeds, mix, row["alpha"]), rule.features))
        assert {k: row[k] for k in expected} == pytest.approx(expected)
