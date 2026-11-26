"""App MixNext (spec §8): bài đang phát, hàng đợi 10 bài, thanh trượt α. Prototype trên dữ liệu giả (#11).

    python -m src.data.fake --out data/fake   # một lần
    streamlit run app/main.py                 # ở thư mục gốc repo, để Streamlit đọc .streamlit/config.toml (chạy offline)

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

FAKE_DATA_DIR = "data/fake"
FAKE_DATA_COMMAND = f"python -m src.data.fake --out {FAKE_DATA_DIR}"
DEFAULT_ALPHA = 0.5
TWO_DECIMALS = st.column_config.NumberColumn(format="%.2f")


@st.cache_resource(max_entries=1)
def load_player(cfg: AppConfig, modified: tuple[int, ...]) -> Player:
    """Đọc lại khi file dữ liệu đổi (modified: thời điểm sửa của từng file), không cần khởi động lại app."""
    del modified  # chỉ để làm khóa cache; không đặt tên _modified vì Streamlit bỏ tham số có dấu _ khỏi khóa
    return Player.from_config(cfg)


def stop_with_error(message: str) -> None:
    st.error(message)
    st.stop()


def missing_files_message(missing: list[Path], config_path: Path) -> str:
    """Chỉ nhắc lệnh sinh dữ liệu giả khi file thiếu nằm trong data/fake."""
    fix = f"sửa đường dẫn trong `{config_path}`"
    if any(FAKE_DATA_DIR in p.as_posix() for p in missing):
        fix = f"tạo dữ liệu giả bằng `{FAKE_DATA_COMMAND}` (chạy ở thư mục gốc repo), hoặc {fix}"
    return "Không tìm thấy file dữ liệu: " + ", ".join(f"`{p}`" for p in missing) + f".\n\nCách sửa: {fix}."


st.set_page_config(page_title="MixNext", layout="wide")
st.title("MixNext")
st.caption("Gợi ý bài tiếp theo vừa hợp gu vừa chuyển bài mượt. Prototype trên dữ liệu giả (#11).")

config_path = Path(os.environ.get("MIXNEXT_CONFIG", DEFAULT_CONFIG))
try:
    cfg = load_config(config_path)
except (OSError, ValueError) as err:  # không có file config, TOML sai (TOMLDecodeError là ValueError), thiếu khóa
    stop_with_error(f"Không đọc được file config `{config_path}`: {err}")
missing = cfg.missing_files()
if missing:
    stop_with_error(missing_files_message(missing, config_path))
try:
    player = load_player(cfg, tuple(p.stat().st_mtime_ns for p in cfg.files))
except ValueError as err:  # SchemaError: dữ liệu không đúng docs/data-contract.md
    stop_with_error(f"Dữ liệu không đúng định dạng: {err}")

with st.sidebar:
    session_id = st.selectbox(f"Playlist (phiên {player.split})", player.playlists, format_func=lambda s: f"Playlist {s}")
    alpha = st.slider("α", 0.0, 1.0, DEFAULT_ALPHA, 0.05, help="α = 1: chỉ theo điểm gu. α = 0: chỉ theo điểm mix.")
    st.caption("← mix mượt · đúng gu →")
    st.caption(f"Ứng viên: `{cfg.candidates.name}`")

playing = player.now_playing(session_id)
st.subheader("Bài đang phát")
st.markdown(f"**{playing['track_name']}** · {playing['artist_name']}")
bpm, camelot, energy = st.columns(3)
bpm.metric("BPM", f"{playing['bpm']:.0f}")
camelot.metric("Camelot", playing["camelot"])
energy.metric("Năng lượng", f"{playing['energy']:.2f}")
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
    column_config={
        "BPM": st.column_config.NumberColumn(format="%.0f"),
        "Năng lượng": TWO_DECIMALS,
        "Điểm gu": TWO_DECIMALS,
        "Điểm mix": TWO_DECIMALS,
        "Điểm cuối": TWO_DECIMALS,
    },
)
