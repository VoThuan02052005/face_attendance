"""
app.py - Hệ thống chấm công bằng nhận diện khuôn mặt
Streamlit UI với 3 trang: Chấm công / Đăng ký / Lịch sử
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
    get_attendance_history,
    get_attendance_today,
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

# ── CSS tuỳ chỉnh giao diện ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Nền tổng thể */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* Card container */
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Tiêu đề chính */
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-card .metric-value {
    font-size: 2rem; font-weight: 700;
    color: #a78bfa;
}
.metric-card .metric-label {
    font-size: 0.78rem; color: #94a3b8; margin-top: 0.2rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.25s ease;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(102,126,234,0.45);
}

/* Input fields */
.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stDateInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
    color: white !important;
}

/* Alert boxes */
.stSuccess, .stError, .stWarning, .stInfo {
    border-radius: 10px !important;
}

/* DataFrame */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Divider */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* Section headers */
h2, h3 { color: #e2e8f0 !important; }

/* Upload area */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 2px dashed rgba(167,139,250,0.4) !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Khởi tạo DB ────────────────────────────────────────────────────────────────
init_db()

# ── Cache: load known faces (reload khi cần) ────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_known_faces():
    return load_known_faces()

# ── Sidebar Navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Face Attendance")
    st.markdown("---")
    page = st.radio(
        "Điều hướng",
        ["🏠 Chấm công", "👤 Đăng ký nhân viên", "📋 Lịch sử chấm công"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Stats nhanh
    employees = get_all_employees()
    today_records = get_attendance_today()
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-value">{len(employees)}</div>
        <div class="metric-label">Nhân viên</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{len(today_records)}</div>
        <div class="metric-label">Hôm nay</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"📅 {date.today().strftime('%d/%m/%Y')}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TRANG 1: CHẤM CÔNG                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
if page == "🏠 Chấm công":
    st.markdown('<div class="hero-title">🏠 Chấm công Tự động</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Chụp ảnh để hệ thống nhận diện khuôn mặt và ghi chấm công</div>', unsafe_allow_html=True)

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
                # Đọc ảnh
                pil_img = Image.open(camera_img).convert("RGB")
                frame_rgb = np.array(pil_img)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # Load known faces
                known_encodings, known_names = cached_known_faces()

                if not known_encodings:
                    st.warning("⚠️ Chưa có nhân viên nào được đăng ký.\nVui lòng đăng ký nhân viên trước.")
                else:
                    results = recognize_faces(frame_rgb, known_encodings, known_names)
                    annotated = draw_results(frame_bgr.copy(), results)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                    st.image(annotated_rgb, use_container_width=True)

                    if not results:
                        st.warning("😕 Không phát hiện khuôn mặt nào trong ảnh.")
                    else:
                        st.markdown("---")
                        for res in results:
                            name = res["name"]
                            if name == "Không nhận diện được":
                                st.error(f"❓ **{name}** – Khuôn mặt không có trong hệ thống")
                            else:
                                saved = record_attendance(name)
                                if saved:
                                    st.success(f"✅ **{name}** – Chấm công thành công!")
                                    st.balloons()
                                else:
                                    st.info(f"ℹ️ **{name}** – Đã chấm công hôm nay rồi.")
        else:
            st.markdown("""
            <div class="card" style="text-align:center; padding: 2.5rem 1rem; color: #94a3b8;">
                <div style="font-size:3rem; margin-bottom:0.75rem;">📷</div>
                <div style="font-size:1rem; font-weight:500;">Bấm nút chụp ở bên trái<br>để nhận diện khuôn mặt</div>
            </div>
            """, unsafe_allow_html=True)

    # Bảng chấm công hôm nay
    st.markdown("---")
    st.markdown("### 📊 Chấm công hôm nay")
    today_records = get_attendance_today()
    if today_records:
        df_today = pd.DataFrame(today_records)[["name", "timestamp"]]
        df_today.columns = ["Nhân viên", "Thời gian"]
        df_today.index = range(1, len(df_today) + 1)
        st.dataframe(df_today, use_container_width=True)
    else:
        st.info("Chưa có chấm công nào hôm nay.")


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
            name_input = st.text_input(
                "Họ và tên nhân viên *",
                placeholder="Ví dụ: Nguyen Van A",
            )
            uploaded_file = st.file_uploader(
                "Ảnh chân dung *",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Chọn ảnh rõ mặt, ánh sáng tốt, 1 người/ảnh",
            )

            if uploaded_file:
                preview_img = Image.open(uploaded_file).convert("RGB")
                st.image(preview_img, caption="Xem trước ảnh", use_container_width=True)

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
                        # Lưu ảnh vào dataset/employees
                        img_path = save_employee_image(name_input.strip(), pil_img)

                        # Lưu vào DB
                        ok = add_employee(name_input.strip(), img_path)
                        if ok:
                            # Xoá cache face encodings để reload
                            cached_known_faces.clear()
                            st.success(f"✅ Đã đăng ký **{name_input.strip()}** thành công!")
                        else:
                            st.warning(f"⚠️ Tên **{name_input.strip()}** đã tồn tại trong hệ thống.")

    with col_list:
        st.markdown("### 📋 Danh sách nhân viên")
        employees = get_all_employees()

        if employees:
            for emp in employees:
                with st.container():
                    c1, c2, c3 = st.columns([2, 2, 1])
                    with c1:
                        st.markdown(f"**{emp['name']}**")
                    with c2:
                        st.caption(emp.get("created_at", ""))
                    with c3:
                        if st.button("🗑️", key=f"del_{emp['id']}", help="Xóa nhân viên"):
                            import os
                            # Xóa ảnh
                            try:
                                os.remove(emp["image_path"])
                            except FileNotFoundError:
                                pass
                            delete_employee(emp["name"])
                            cached_known_faces.clear()
                            st.rerun()
                    st.divider()
        else:
            st.markdown("""
            <div class="card" style="text-align:center; color:#94a3b8; padding:2rem;">
                <div style="font-size:2.5rem">👥</div>
                <div>Chưa có nhân viên nào.<br>Hãy thêm nhân viên đầu tiên!</div>
            </div>
            """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TRANG 3: LỊCH SỬ CHẤM CÔNG                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
elif page == "📋 Lịch sử chấm công":
    st.markdown('<div class="hero-title">📋 Lịch sử Chấm công</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Xem và xuất báo cáo lịch sử chấm công</div>', unsafe_allow_html=True)

    filter_col, export_col = st.columns([2, 1], gap="large")

    with filter_col:
        filter_mode = st.radio(
            "Lọc theo",
            ["Tất cả", "Theo ngày"],
            horizontal=True,
            label_visibility="collapsed",
        )

    selected_date = None
    if filter_mode == "Theo ngày":
        with filter_col:
            selected_date = st.date_input("Chọn ngày", value=date.today())

    # Lấy dữ liệu
    date_str = selected_date.isoformat() if selected_date else None
    records = get_attendance_history(filter_date=date_str)

    if records:
        df = pd.DataFrame(records)[["name", "timestamp", "date"]]
        df.columns = ["Nhân viên", "Thời gian chấm công", "Ngày"]
        df.index = range(1, len(df) + 1)

        # Metrics tổng quan
        total = len(df)
        unique_emp = df["Nhân viên"].nunique()
        unique_days = df["Ngày"].nunique()

        m1, m2, m3 = st.columns(3)
        m1.metric("📌 Tổng bản ghi", total)
        m2.metric("👥 Nhân viên", unique_emp)
        m3.metric("📅 Số ngày", unique_days)

        st.markdown("---")
        st.dataframe(df, use_container_width=True, height=420)

        # Export CSV
        with export_col:
            csv_data = df.to_csv(index=False, encoding="utf-8-sig")
            filename = f"chamcong_{date_str or 'all'}.csv"
            st.download_button(
                label="⬇️ Xuất CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                use_container_width=True,
            )
    else:
        label = f"ngày {selected_date.strftime('%d/%m/%Y')}" if selected_date else "hệ thống"
        st.info(f"📭 Chưa có dữ liệu chấm công trong {label}.")
