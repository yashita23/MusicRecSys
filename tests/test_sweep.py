import numpy as np
import pandas as pd
import pytest

from src.data.fake import make_fake_dataset, write_fake
from src.eval.run import evaluate
from src.eval.sweep import ALPHAS, choose_alpha, main, sweep
from src.models.item_knn import ItemKNN
from src.rerank.generate import CHAIN_LENGTH
from src.rerank.mix_rule import RuleMix

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def is_png(path) -> bool:
    return path.read_bytes()[:8] == PNG_MAGIC


@pytest.fixture(scope="module")
def fake():
    return make_fake_dataset(n_sessions=200, n_tracks=400, random_seed=4)


@pytest.fixture(scope="module")
def table(fake):
    model = ItemKNN().fit(fake.sessions)
    mix = RuleMix(fake.tracks)
    return sweep(fake.candidates["val"], fake.eval_targets, model, mix, mix.features, n_smooth=15, random_seed=0)


def test_one_row_per_alpha_from_0_to_1_in_steps_of_0_1(table):
    np.testing.assert_allclose(table["alpha"], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert list(table["alpha"]) == list(ALPHAS)


def test_alpha_one_keeps_accuracy_of_the_original_candidates(fake, table):
    original = evaluate(fake.candidates["val"], fake.eval_targets)
    row = table.set_index("alpha").loc[1.0]
    assert (row["hit_rate"], row["ndcg"], row["mrr"]) == pytest.approx((original.hit_rate, original.ndcg, original.mrr))
    assert row["n_sessions"] == original.n_sessions


def test_pure_mix_is_at_least_as_smooth_as_pure_taste(table):
    by_alpha = table.set_index("alpha")
    assert by_alpha.loc[0.0, "mixable_pct"] >= by_alpha.loc[1.0, "mixable_pct"]


def test_smoothness_uses_the_same_sample_at_every_alpha(table):
    assert (table["n_pairs"] == 15 * CHAIN_LENGTH).all()  # 15 chuỗi × 10 cặp ở mọi α


def test_choose_alpha_takes_most_mixable_alpha_within_10_percent_ndcg_drop():
    # α = 1: NDCG 0,10 → ngưỡng 0,09. Đạt ngưỡng: α 1, 0,8 (0,095), 0,6 (0,09). Trong đó α 0,6 có % cặp mix được cao nhất.
    candidates = pd.DataFrame({"alpha": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                               "ndcg": [0.05, 0.07, 0.089, 0.09, 0.095, 0.10],
                               "mixable_pct": [80.0, 70.0, 60.0, 40.0, 10.0, 1.0]})
    assert choose_alpha(candidates) == 0.6


def test_cli_writes_table_and_tradeoff_figure(tmp_path):
    data = tmp_path / "data"
    write_fake(make_fake_dataset(n_sessions=120, n_tracks=300, random_seed=5), data)
    out = tmp_path / "reports"
    main(["--data", str(data), "--candidates", str(data / "candidates" / "fake_val_run0.parquet"),
          "--model", "itemknn", "--out", str(out), "--n-smooth", "10", "--tag", "v1"])
    written = pd.read_csv(out / "sweep_v1.csv")
    assert len(written) == len(ALPHAS)
    assert {"alpha", "ndcg", "hit_rate", "mrr", "mixable_pct", "bpm_delta_pct", "key_compatible_pct", "energy_gap"} <= set(written.columns)
    assert written["model"].unique().tolist() == ["itemknn"]
    assert written["mix_weights"].unique().tolist() == ["0.4/0.4/0.2"]
    assert is_png(out / "fig_tradeoff_v1.png")


def test_cli_rejects_candidates_from_another_model(tmp_path):
    data = tmp_path / "data"
    write_fake(make_fake_dataset(n_sessions=60, n_tracks=300, random_seed=5), data)
    itemknn_file = data / "candidates" / "itemknn_val_run0.parquet"
    (data / "candidates" / "fake_val_run0.parquet").rename(itemknn_file)
    with pytest.raises(SystemExit):
        main(["--data", str(data), "--candidates", str(itemknn_file), "--model", "popularity", "--out", str(tmp_path / "r")])


def test_cli_plot_only_redraws_figure_from_csv(tmp_path):
    csv = tmp_path / "sweep_v1.csv"
    pd.DataFrame({"alpha": [0.0, 0.5, 1.0], "ndcg": [0.1, 0.12, 0.13], "mixable_pct": [60.0, 20.0, 2.0],
                  "model": "itemknn", "mix_weights": "0.4/0.4/0.2"}).to_csv(csv, index=False)
    main(["--plot-only", str(csv), "--out", str(tmp_path / "reports"), "--tag", "v1"])
    assert is_png(tmp_path / "reports" / "fig_tradeoff_v1.png")
