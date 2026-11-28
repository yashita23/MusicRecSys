"""App MixNext, Demo v1 (#21, spec §8 mục 1–3, 5): chọn playlist test, bài đang phát, hàng đợi top-10 kèm giải thích,
thanh trượt α, người dùng chọn bài tiếp theo, lịch sử phiên, quay lại, bắt đầu lại.

    streamlit run app/main.py      # ở thư mục gốc repo, để Streamlit đọc .streamlit/config.toml (chạy offline)

Đọc artifacts/ (#20) theo app/config.toml; muốn dùng file config khác thì đặt biến môi trường MIXNEXT_CONFIG.
App chỉ gợi ý: không tự phát và không tự chuyển bài. Mọi logic nằm trong app/player.py; file này chỉ hiển thị.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `streamlit run` chỉ thêm app/ vào sys.path; cần gốc repo để import app và src
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.config import DEFAULT_CONFIG, AppConfig, load_config  # noqa: E402
from app.player import Player  # noqa: E402

BUILD_COMMANDS = (
    "python -m src.export_artifacts --input <thư mục Kaggle đã tải> --out artifacts   # bản thật (#20)\n"
    "python -m src.export_artifacts --fake --out artifacts                          # dữ liệu giả để thử"
)
# (nhãn, độ rộng) của các cột hàng đợi
QUEUE_COLUMNS = (("#", 0.4), ("Bài", 3.2), ("BPM", 0.8), ("Camelot", 0.8), ("Năng lượng", 0.9), ("Điểm gu", 0.8),
                 ("Điểm mix", 0.8), ("Điểm cuối", 0.8), ("Giải thích", 4.0), ("", 0.8))
WIDTHS = [width for _, width in QUEUE_COLUMNS]


def decimal(x: float) -> str:
    """Số 2 chữ số thập phân với dấu phẩy, như dòng giải thích."""
    return f"{x:.2f}".replace(".", ",")


@st.cache_resource(max_entries=1)
def load_player(cfg: AppConfig, modified: tuple[int, ...]) -> Player:
    """Đọc lại khi file artifacts đổi (modified: thời điểm sửa của từng file), không cần khởi động lại app."""
    del modified  # chỉ để làm khóa cache; không đặt tên _modified vì Streamlit bỏ tham số có dấu _ khỏi khóa
    return Player.from_dir(cfg.artifacts)


def stop_with_error(message: str) -> None:
    st.error(message)
    st.stop()


st.set_page_config(page_title="MixNext", layout="wide")
st.title("MixNext")
st.caption("Gợi ý bài tiếp theo vừa hợp gu vừa chuyển bài mượt. Bạn chọn bài tiếp theo trong hàng đợi; app không tự phát.")

config_path = Path(os.environ.get("MIXNEXT_CONFIG", DEFAULT_CONFIG))
try:
    cfg = load_config(config_path)
except (OSError, ValueError) as err:  # không có file config, TOML sai (TOMLDecodeError là ValueError), thiếu khóa
    stop_with_error(f"Không đọc được file config `{config_path}`: {err}")
missing = cfg.missing_files()
if missing:
    stop_with_error(
        f"Thiếu file trong `{cfg.artifacts}`: " + ", ".join(f"`{p.name}`" for p in missing)
        + f".\n\nTạo artifacts/ (chạy ở thư mục gốc repo):\n```\n{BUILD_COMMANDS}\n```\nhoặc sửa đường dẫn trong `{config_path}`."
    )
try:
    player = load_player(cfg, tuple(p.stat().st_mtime_ns for p in cfg.files))
except ValueError as err:  # artifacts sai định dạng (load_artifacts kiểm tra)
    stop_with_error(f"artifacts/ không đúng định dạng: {err}")

with st.sidebar:
    playlist = st.selectbox("Playlist (phiên test)", player.playlists, format_func=lambda s: f"Playlist {s}", key="playlist")
    alpha = st.slider("α", 0.0, 1.0, float(player.default_alpha), 0.05, key="alpha",
                      help="α = 1: chỉ theo điểm gu. α = 0: chỉ theo điểm mix.")
    st.caption("← mix mượt · đúng gu →")
    current = st.session_state.get("session")
    if current is None or current.playlist_id != playlist or not player.is_valid(current):  # đổi playlist hoặc đổi artifacts
        st.session_state["session"] = player.start(playlist)
    back, restart = st.columns(2)
    if back.button("Quay lại", key="back", use_container_width=True):
        st.session_state["session"] = player.back(st.session_state["session"])
    if restart.button("Bắt đầu lại", key="restart", use_container_width=True):
        st.session_state["session"] = player.restart(st.session_state["session"])
    st.caption(f"Mô hình gu: {player.model} · mix theo luật")

session = st.session_state["session"]
queue = player.queue(session, alpha)
for track_id in queue["track_id"]:  # nút chọn xử lý trước khi vẽ, để bài đang phát và hàng đợi cập nhật ngay
    if st.session_state.get(f"choose_{track_id}"):
        session = st.session_state["session"] = player.choose(session, track_id, alpha)
        queue = player.queue(session, alpha)
        break

playing = player.now_playing(session)
st.subheader("Bài đang phát")
st.markdown(f"**{playing['track_name']}** · {playing['artist_name']}")
bpm, camelot, energy, step = st.columns(4)
bpm.metric("BPM", f"{playing['bpm']:.0f}")
camelot.metric("Camelot", playing["camelot"])
energy.metric("Năng lượng", decimal(playing["energy"]))
step.metric("Bài đã chọn", len(session.chosen))

st.subheader("Hàng đợi gợi ý")
for col, (label, _) in zip(st.columns(WIDTHS), QUEUE_COLUMNS):
    col.markdown(f"**{label}**")
for row in queue.itertuples():
    cols = st.columns(WIDTHS)
    cols[0].write(row.rank)
    cols[1].write(f"{row.track_name} · {row.artist_name}")
    cols[2].write(f"{row.bpm:.0f}")
    cols[3].write(row.camelot)
    cols[4].write(decimal(row.energy))
    cols[5].write(decimal(row.gu_norm))
    cols[6].write(decimal(row.mix))
    cols[7].write(decimal(row.final))
    cols[8].write(row.explanation)
    cols[9].button("Chọn", key=f"choose_{row.track_id}")

st.subheader("Lịch sử phiên")
history = player.history(session)
st.dataframe(
    history[["step", "source", "track_id", "track_name", "artist_name", "bpm", "camelot", "energy", "explanation"]],
    hide_index=True,
    column_config={
        "step": "Bước", "source": "Nguồn", "track_id": None, "track_name": "Bài", "artist_name": "Nghệ sĩ",
        "bpm": st.column_config.NumberColumn("BPM", format="%.0f"), "camelot": "Camelot",
        "energy": st.column_config.NumberColumn("Năng lượng", format="%.2f"), "explanation": "Chuyển từ bài trước",
    },
)
