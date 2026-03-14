"""
app.py - Hệ thống chấm công đa phiên (Multi-Session) + Báo cáo tuần
Streamlit UI với 4 trang: Chấm công / Đăng ký / Lịch sử / Báo cáo tuần
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from datetime import date

from utils.db_utils import (
    init_db,
    add_employee,
    get_all_employees,
    delete_employee,
    record_attendance,
    get_sessions_today,
    get_sessions_by_date,
    get_weekly_summary,
    get_daily_summary_this_week,
    current_week_range,
    today_vn,
)
from utils.face_utils import (
    load_known_faces,
    recognize_faces,
    draw_results,
    save_employee_image,
    validate_face_image,
)

# ── Cấu hình trang ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hệ thống Chấm công AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS tùy chỉnh ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

.hero-title {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.2rem;
}
.hero-sub { color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.2rem; }

.metric-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 120px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px; padding: 0.9rem 1.1rem; text-align: center;
}
.metric-card .metric-value { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
.metric-card .metric-label { font-size: 0.76rem; color: #94a3b8; margin-top: 0.2rem; }

.session-badge {
    display: inline-block;
    padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600;
}
.badge-in  { background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid #34d399; }
.badge-out { background: rgba(248,113,113,0.2); color: #f87171; border: 1px solid #f87171; }
.badge-open{ background: rgba(251,191,36,0.2);  color: #fbbf24; border: 1px solid #fbbf24; }

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important; border: none; border-radius: 10px;
    padding: 0.55rem 1.2rem; font-weight: 600; font-size: 0.88rem;
    transition: all 0.25s ease; width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(102,126,234,0.45);
}
.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stDateInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important; color: white !important;
}
.stSuccess, .stError, .stWarning, .stInfo { border-radius: 10px !important; }
.stDataFrame { border-radius: 12px; overflow: hidden; }
hr { border-color: rgba(255,255,255,0.08) !important; }
h2, h3 { color: #e2e8f0 !important; }
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 2px dashed rgba(167,139,250,0.4) !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Khởi tạo DB ───────────────────────────────────────────────────────────────
init_db()

# ── Cache model InsightFace ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Đang tải model nhận diện khuôn mặt...")
def cached_known_faces():
    return load_known_faces()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Face Attendance")
    st.markdown("---")
    page = st.radio(
        "nav",
        ["🏠 Chấm công", "👤 Đăng ký nhân viên", "📋 Lịch sử", "📊 Báo cáo tuần"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Thống kê nhanh
    employees = get_all_employees()
    sessions_today = get_sessions_today()
    checked_today = len({s["name"] for s in sessions_today})

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-value">{len(employees)}</div>
        <div class="metric-label">Nhân viên</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{checked_today}</div>
        <div class="metric-label">Hôm nay</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    monday, sunday = current_week_range()
    st.caption(f"📅 {today_vn()} (VN)")
    st.caption(f"📆 Tuần: {monday} → {sunday}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TRANG 1: CHẤM CÔNG                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
if page == "🏠 Chấm công":
    st.markdown('<div class="hero-title">🏠 Chấm công Tự động</div>', unsafe_allow_html=True)
    st.markdown("""<div class="hero-sub">
        📷 Quét lẻ (1, 3…) = Giờ <b style="color:#34d399">VÀO</b> &nbsp;|&nbsp;
        📷 Quét chẵn (2, 4…) = Giờ <b style="color:#f87171">RA</b>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.markdown("### 📸 Chụp ảnh từ Camera")
        camera_img = st.camera_input(
            label="Nhìn thẳng vào camera và bấm chụp",
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("### 🔍 Kết quả nhận diện")

        if camera_img is not None:
            with st.spinner("Đang nhận diện khuôn mặt..."):
                pil_img = Image.open(camera_img).convert("RGB")
                frame_rgb = np.array(pil_img)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                known_encodings, known_names = cached_known_faces()

                if not known_encodings:
                    st.warning("⚠️ Chưa có nhân viên nào được đăng ký.")
                else:
                    results = recognize_faces(frame_rgb, known_encodings, known_names)
                    annotated = draw_results(frame_bgr.copy(), results)
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                    if not results:
                        st.warning("😕 Không phát hiện khuôn mặt nào trong ảnh.")
                    else:
                        st.markdown("---")
                        for res in results:
                            name = res["name"]
                            if name == "Không nhận diện được":
                                st.error(f"❓ **{name}** – Không có trong hệ thống")
                            else:
                                att = record_attendance(name)
                                sn = att["session_num"]
                                if att["status"] == "checked_in":
                                    st.success(
                                        f"🟢 **{name}** – Phiên {sn}: "
                                        f"VÀO lúc **{att['check_in']}** (giờ VN)"
                                    )
                                    st.balloons()
                                else:
                                    h = int(att["duration_hours"])
                                    m = int((att["duration_hours"] - h) * 60)
                                    st.success(
                                        f"🔴 **{name}** – Phiên {sn}: "
                                        f"RA lúc **{att['check_out']}** · "
                                        f"Thời gian phiên: **{h}h{m:02d}m**"
                                    )
                                    st.balloons()
        else:
            st.markdown("""
            <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);
                border-radius:16px;padding:2.5rem 1rem;text-align:center;color:#94a3b8;">
                <div style="font-size:3rem;margin-bottom:0.75rem">📷</div>
                <div>Bấm nút chụp ở bên trái để nhận diện khuôn mặt</div>
            </div>""", unsafe_allow_html=True)

    # ── Bảng phiên hôm nay ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Phiên chấm công hôm nay")
    sessions = get_sessions_today()

    if sessions:
        rows = []
        for s in sessions:
            duration_str = f"{s['duration_hours']:.2f}h" if s.get("duration_hours") is not None else "⏳ Đang làm"
            status_html = (
                '<span class="session-badge badge-out">✅ Hoàn thành</span>'
                if s.get("check_out")
                else '<span class="session-badge badge-open">🟡 Đang làm</span>'
            )
            rows.append({
                "Nhân viên": s["name"],
                "Phiên": s["session_num"],
                "Giờ Vào": s.get("check_in", "—"),
                "Giờ Ra": s.get("check_out") or "—",
                "Thời gian": duration_str,
            })

        df = pd.DataFrame(rows)
        df.index = range(1, len(df) + 1)
        st.dataframe(df, use_container_width=True)

        # Tổng giờ mỗi người hôm nay
        total_by_name = {}
        for s in sessions:
            if s.get("duration_hours"):
                total_by_name[s["name"]] = total_by_name.get(s["name"], 0) + s["duration_hours"]
        if total_by_name:
            st.markdown("**Tổng giờ hôm nay:**")
            cols = st.columns(min(len(total_by_name), 4))
            for i, (emp, hrs) in enumerate(total_by_name.items()):
                h, m = int(hrs), int((hrs - int(hrs)) * 60)
                cols[i % 4].metric(emp, f"{h}h{m:02d}m")
    else:
        st.info("Chưa có phiên chấm công nào hôm nay.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TRANG 2: ĐĂNG KÝ NHÂN VIÊN                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
elif page == "👤 Đăng ký nhân viên":
    st.markdown('<div class="hero-title">👤 Đăng ký Nhân viên</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Thêm nhân viên mới vào hệ thống bằng cách upload ảnh chân dung</div>', unsafe_allow_html=True)

    col_form, col_list = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("### ➕ Thêm nhân viên mới")
        with st.form("register_form", clear_on_submit=True):
            name_input = st.text_input("Họ và tên nhân viên *", placeholder="Ví dụ: Nguyen Van A")
            uploaded_file = st.file_uploader(
                "Ảnh chân dung *",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Chọn ảnh rõ mặt, ánh sáng tốt, 1 người/ảnh",
            )
            if uploaded_file:
                st.image(Image.open(uploaded_file).convert("RGB"), caption="Xem trước", use_container_width=True)

            submitted = st.form_submit_button("💾 Đăng ký nhân viên", use_container_width=True)
            if submitted:
                if not name_input.strip():
                    st.error("❌ Vui lòng nhập tên nhân viên.")
                elif uploaded_file is None:
                    st.error("❌ Vui lòng upload ảnh chân dung.")
                else:
                    pil_img = Image.open(uploaded_file).convert("RGB")
                    with st.spinner("Đang kiểm tra ảnh..."):
                        valid, msg = validate_face_image(pil_img)
                    if not valid:
                        st.error(f"❌ {msg}")
                    else:
                        img_path = save_employee_image(name_input.strip(), pil_img)
                        if add_employee(name_input.strip(), img_path):
                            cached_known_faces.clear()
                            st.success(f"✅ Đã đăng ký **{name_input.strip()}** thành công!")
                        else:
                            st.warning(f"⚠️ Tên **{name_input.strip()}** đã tồn tại.")

    with col_list:
        st.markdown("### 📋 Danh sách nhân viên")
        employees = get_all_employees()
        if employees:
            for emp in employees:
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    st.markdown(f"**{emp['name']}**")
                with c2:
                    st.caption(emp.get("created_at", ""))
                with c3:
                    if st.button("🗑️", key=f"del_{emp['id']}", help="Xóa"):
                        import os as _os
                        try:
                            _os.remove(emp["image_path"])
                        except FileNotFoundError:
                            pass
                        delete_employee(emp["name"])
                        cached_known_faces.clear()
                        st.rerun()
                st.divider()
        else:
            st.info("Chưa có nhân viên nào. Hãy thêm nhân viên đầu tiên!")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TRANG 3: LỊCH SỬ CHẤM CÔNG                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
elif page == "📋 Lịch sử":
    st.markdown('<div class="hero-title">📋 Lịch sử Chấm công</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Xem toàn bộ lịch sử các phiên chấm công và xuất báo cáo</div>', unsafe_allow_html=True)

    filter_col, export_col = st.columns([2, 1], gap="large")
    with filter_col:
        filter_mode = st.radio("Lọc", ["Tất cả", "Theo ngày"], horizontal=True, label_visibility="collapsed")

    selected_date = None
    if filter_mode == "Theo ngày":
        with filter_col:
            selected_date = st.date_input("Chọn ngày", value=date.today())

    date_str = selected_date.isoformat() if selected_date else None
    records = get_sessions_by_date(filter_date=date_str)

    if records:
        df = pd.DataFrame(records)
        available = [c for c in ["name", "date", "session_num", "check_in", "check_out", "duration_hours"] if c in df.columns]
        df = df[available].copy()
        df.rename(columns={
            "name": "Nhân viên", "date": "Ngày", "session_num": "Phiên",
            "check_in": "Giờ Vào", "check_out": "Giờ Ra", "duration_hours": "Thời gian (h)"
        }, inplace=True)
        df.index = range(1, len(df) + 1)

        # Metrics
        total_sessions = len(df)
        unique_emp = df["Nhân viên"].nunique()
        unique_days = df["Ngày"].nunique() if "Ngày" in df.columns else 0
        total_hours = df["Thời gian (h)"].sum() if "Thời gian (h)" in df.columns else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📌 Phiên", total_sessions)
        m2.metric("👥 Nhân viên", unique_emp)
        m3.metric("📅 Ngày", unique_days)
        m4.metric("⏱️ Tổng giờ", f"{total_hours:.1f}h")

        st.markdown("---")
        st.dataframe(df, use_container_width=True, height=420)

        with export_col:
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "⬇️ Xuất CSV", csv,
                file_name=f"chamcong_{date_str or 'all'}.csv",
                mime="text/csv", use_container_width=True,
            )
    else:
        label = f"ngày {selected_date.strftime('%d/%m/%Y')}" if selected_date else "hệ thống"
        st.info(f"📭 Chưa có dữ liệu chấm công trong {label}.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TRANG 4: BÁO CÁO TUẦN                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
elif page == "📊 Báo cáo tuần":
    monday, sunday = current_week_range()

    st.markdown('<div class="hero-title">📊 Báo cáo Tuần</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hero-sub">Tổng hợp giờ làm việc trong tuần: '
        f'<b style="color:#a78bfa">{monday}</b> → <b style="color:#a78bfa">{sunday}</b></div>',
        unsafe_allow_html=True,
    )

    weekly = get_weekly_summary()
    daily_detail = get_daily_summary_this_week()

    # ── Tổng quan toàn công ty ────────────────────────────────────────────────
    if weekly:
        total_emp_worked = len(weekly)
        total_hrs_all = sum(r["total_hours"] for r in weekly)
        total_sessions_all = sum(r["sessions_count"] for r in weekly)

        m1, m2, m3 = st.columns(3)
        m1.metric("👥 Nhân viên đi làm", total_emp_worked)
        m2.metric("⏱️ Tổng giờ toàn công ty", f"{total_hrs_all:.1f}h")
        m3.metric("🔄 Tổng phiên", total_sessions_all)

        st.markdown("---")

        # ── Bảng tổng giờ từng nhân viên ────────────────────────────────────
        st.markdown("### 🏆 Tổng giờ làm – Từng nhân viên")
        df_weekly = pd.DataFrame(weekly)
        df_weekly.rename(columns={
            "name": "Nhân viên",
            "total_hours": "Tổng giờ (h)",
            "sessions_count": "Số phiên",
            "days_worked": "Ngày đi làm",
        }, inplace=True)

        # Thêm cột giờ dạng HH:MM
        def to_hhmm(h):
            return f"{int(h)}h{int((h - int(h)) * 60):02d}m"

        df_weekly["Tổng (HH:MM)"] = df_weekly["Tổng giờ (h)"].apply(to_hhmm)
        df_weekly.index = range(1, len(df_weekly) + 1)
        st.dataframe(df_weekly[["Nhân viên", "Ngày đi làm", "Số phiên", "Tổng giờ (h)", "Tổng (HH:MM)"]], use_container_width=True)

        # ── Chi tiết theo ngày ───────────────────────────────────────────────
        if daily_detail:
            st.markdown("---")
            st.markdown("### 📅 Chi tiết từng ngày trong tuần")

            df_daily = pd.DataFrame(daily_detail)
            df_daily.rename(columns={
                "name": "Nhân viên",
                "date": "Ngày",
                "total_hours": "Giờ làm (h)",
                "sessions_count": "Phiên",
            }, inplace=True)
            df_daily["Giờ làm"] = df_daily["Giờ làm (h)"].apply(to_hhmm)
            df_daily.index = range(1, len(df_daily) + 1)
            st.dataframe(df_daily[["Ngày", "Nhân viên", "Phiên", "Giờ làm (h)", "Giờ làm"]], use_container_width=True)

        # ── Export ───────────────────────────────────────────────────────────
        st.markdown("---")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            csv_w = df_weekly.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "⬇️ Xuất tổng hợp tuần (CSV)",
                csv_w,
                file_name=f"bao_cao_tuan_{monday}_{sunday}.csv",
                mime="text/csv", use_container_width=True,
            )
        if daily_detail:
            with col_e2:
                csv_d = df_daily.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    "⬇️ Xuất chi tiết ngày (CSV)",
                    csv_d,
                    file_name=f"chi_tiet_tuan_{monday}_{sunday}.csv",
                    mime="text/csv", use_container_width=True,
                )
    else:
        st.info(f"📭 Chưa có dữ liệu chấm công trong tuần này ({monday} → {sunday}).")
        st.markdown("""
        <div style="background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.2);
            border-radius:12px;padding:1.5rem;color:#94a3b8;text-align:center;">
            <div style="font-size:2.5rem;margin-bottom:0.5rem">📅</div>
            <div>Nhân viên cần chấm công ít nhất 1 lần trong tuần này để hiển thị báo cáo.</div>
        </div>""", unsafe_allow_html=True)
