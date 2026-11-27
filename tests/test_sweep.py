import numpy as np
import pytest

from src.data.fake import make_fake_dataset
from src.eval.run import evaluate
from src.eval.sweep import ALPHAS, sweep
from src.models.item_knn import ItemKNN
from src.rerank.mix_rule import RuleMix


@pytest.fixture(scope="module")
def fake():
    return make_fake_dataset(n_sessions=200, n_tracks=400, random_seed=4)


@pytest.fixture(scope="module")
def table(fake):
    model = ItemKNN().fit(fake.sessions[fake.sessions["split"] == "train"])
    mix = RuleMix(fake.tracks)
    return sweep(fake.candidates["val"], fake.eval_targets, model, mix, mix.features, n_smooth=15, random_seed=0)


def test_one_row_per_alpha_from_0_to_1_in_steps_of_0_1(table):
    np.testing.assert_allclose(table["alpha"], [round(0.1 * i, 1) for i in range(11)])
    assert list(ALPHAS) == list(table["alpha"])


def test_alpha_one_keeps_accuracy_of_the_original_candidates(fake, table):
    original = evaluate(fake.candidates["val"], fake.eval_targets)
    row = table.set_index("alpha").loc[1.0]
    assert (row["hit_rate"], row["ndcg"], row["mrr"]) == pytest.approx((original.hit_rate, original.ndcg, original.mrr))
    assert row["n_sessions"] == original.n_sessions


def test_pure_mix_is_at_least_as_smooth_as_pure_taste(table):
    by_alpha = table.set_index("alpha")
    assert by_alpha.loc[0.0, "mixable_pct"] >= by_alpha.loc[1.0, "mixable_pct"]
    assert (table["n_pairs"] == 15 * 10).all()  # 15 chuỗi × 10 cặp ở mọi α


def test_cli_writes_table_and_tradeoff_figure(tmp_path):
    import pandas as pd

    from src.data.fake import write_fake
    from src.eval.sweep import main

    data = tmp_path / "data"
    write_fake(make_fake_dataset(n_sessions=120, n_tracks=300, random_seed=5), data)
    out = tmp_path / "reports"
    main(["--data", str(data), "--candidates", str(data / "candidates" / "fake_val_run0.parquet"),
          "--model", "itemknn", "--out", str(out), "--n-smooth", "10", "--tag", "v1"])
    table = pd.read_csv(out / "sweep_v1.csv")
    assert len(table) == 11
    assert {"alpha", "ndcg", "hit_rate", "mrr", "mixable_pct", "bpm_delta_pct", "key_compatible_pct", "energy_gap"} <= set(table.columns)
    assert (out / "fig_tradeoff_v1.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_plot_only_redraws_figure_from_csv(tmp_path):
    import pandas as pd

    from src.eval.sweep import main

    csv = tmp_path / "sweep_v1.csv"
    pd.DataFrame({"alpha": [0.0, 0.5, 1.0], "ndcg": [0.1, 0.12, 0.13], "mixable_pct": [60.0, 20.0, 2.0]}).to_csv(csv, index=False)
    main(["--plot-only", str(csv), "--out", str(tmp_path / "reports"), "--tag", "v1"])
    assert (tmp_path / "reports" / "fig_tradeoff_v1.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
