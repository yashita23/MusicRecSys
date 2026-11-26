"""App MixNext (spec §8): bài đang phát, hàng đợi 10 bài, thanh trượt α. Prototype trên dữ liệu giả (#11).

    python -m src.data.fake --out data/fake   # một lần, ở thư mục gốc repo
    streamlit run app/main.py

Đường dẫn dữ liệu đọc từ app/config.toml; muốn dùng file config khác thì đặt biến môi trường MIXNEXT_CONFIG.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `streamlit run` chỉ thêm app/ vào sys.path; cần gốc repo để import app và src
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.config import DEFAULT_CONFIG, AppConfig, load_config  # noqa: E402
from app.player import Player  # noqa: E402

FAKE_DATA_COMMAND = "python -m src.data.fake --out data/fake"
DEFAULT_ALPHA = 0.5
SCORE = st.column_config.NumberColumn(format="%.2f")


@st.cache_resource
def load_player(cfg: AppConfig) -> Player:
    return Player.from_config(cfg)


st.set_page_config(page_title="MixNext", layout="wide")
st.title("MixNext")
st.caption("Gợi ý bài tiếp theo vừa hợp gu vừa chuyển bài mượt. Prototype trên dữ liệu giả (#11).")

config_path = Path(os.environ.get("MIXNEXT_CONFIG", DEFAULT_CONFIG))
cfg = load_config(config_path)
missing = cfg.missing_files()
if missing:
    st.error(
        "Không tìm thấy file dữ liệu: " + ", ".join(f"`{p}`" for p in missing)
        + f".\n\nTạo dữ liệu giả bằng `{FAKE_DATA_COMMAND}` (chạy ở thư mục gốc repo), hoặc sửa đường dẫn trong `{config_path}`."
    )
    st.stop()
player = load_player(cfg)

with st.sidebar:
    session_id = st.selectbox("Playlist (phiên test)", player.playlists, format_func=lambda s: f"Playlist {s}")
    alpha = st.slider("α", 0.0, 1.0, DEFAULT_ALPHA, 0.05, help="α = 1: chỉ theo điểm gu. α = 0: chỉ theo điểm mix.")
    st.caption("← mix mượt · đúng gu →")
    st.caption(f"Ứng viên: `{cfg.candidates.name}`")

now = player.now_playing(session_id)
st.subheader("Bài đang phát")
st.markdown(f"**{now['track_name']}** · {now['artist_name']}")
bpm, camelot, energy = st.columns(3)
bpm.metric("BPM", f"{now['bpm']:.0f}")
camelot.metric("Camelot", now["camelot"])
energy.metric("Năng lượng", f"{now['energy']:.2f}")
with st.expander("Seed: 5 bài đầu playlist"):
    seed = player.seed(session_id)
    st.markdown("\n".join(f"{i}. {t.track_name} · {t.artist_name} ({t.bpm:.0f} BPM, {t.camelot})" for i, t in enumerate(seed.itertuples(), 1)))

st.subheader("Hàng đợi gợi ý")
queue = player.queue(session_id, alpha)
st.dataframe(
    pd.DataFrame(
        {
            "#": queue["rank"],
            "Bài": queue["track_name"],
            "Nghệ sĩ": queue["artist_name"],
            "BPM": queue["bpm"].round(0),
            "Camelot": queue["camelot"],
            "Năng lượng": queue["energy"],
            "Điểm gu": queue["gu_norm"],
            "Điểm mix": queue["mix"],
            "Điểm cuối": queue["final"],
        }
    ),
    hide_index=True,
    column_config={"BPM": st.column_config.NumberColumn(format="%.0f"), "Năng lượng": SCORE, "Điểm gu": SCORE, "Điểm mix": SCORE, "Điểm cuối": SCORE},
)
