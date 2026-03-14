"""
db_utils.py - Xử lý cơ sở dữ liệu SQLite cho hệ thống chấm công
Múi giờ: Việt Nam (UTC+7) dùng zoneinfo
"""

import sqlite3
import os
from datetime import datetime, date, timezone, timedelta

# ── Múi giờ Việt Nam ──────────────────────────────────────────────────────────
VN_TZ = timezone(timedelta(hours=7))

def now_vn() -> datetime:
    """Trả về thời gian hiện tại theo múi giờ Việt Nam."""
    return datetime.now(VN_TZ)

def today_vn() -> str:
    """Trả về ngày hôm nay theo VN (YYYY-MM-DD)."""
    return now_vn().date().isoformat()

# ── Đường dẫn DB ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "attendance.db")


def get_connection():
    """Trả về kết nối SQLite."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Khởi tạo database và tạo các bảng nếu chưa tồn tại."""
    conn = get_connection()
    cursor = conn.cursor()

    # Bảng employees
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL UNIQUE,
            image_path TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)

    # Bảng attendance (schema mới: check_in + check_out + total_hours)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            date        TEXT NOT NULL,
            check_in    TEXT,
            check_out   TEXT,
            total_hours REAL
        )
    """)

    # Migration: nếu bảng cũ có cột timestamp → thêm cột mới
    existing = {
        row[1]
        for row in cursor.execute("PRAGMA table_info(attendance)").fetchall()
    }
    if "timestamp" in existing and "check_in" not in existing:
        cursor.execute("ALTER TABLE attendance ADD COLUMN check_in TEXT")
        cursor.execute("ALTER TABLE attendance ADD COLUMN check_out TEXT")
        cursor.execute("ALTER TABLE attendance ADD COLUMN total_hours REAL")
        # Chuyển dữ liệu cũ: timestamp → check_in
        cursor.execute("UPDATE attendance SET check_in = timestamp WHERE check_in IS NULL")

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Employees
# ---------------------------------------------------------------------------

def add_employee(name: str, image_path: str) -> bool:
    """Thêm nhân viên. Trả về True nếu thành công, False nếu trùng tên."""
    try:
        conn = get_connection()
        conn.execute(
            "INSERT INTO employees (name, image_path) VALUES (?, ?)",
            (name, image_path),
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def get_all_employees() -> list[dict]:
    """Lấy danh sách tất cả nhân viên."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM employees ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_employee(name: str):
    """Xóa nhân viên theo tên."""
    conn = get_connection()
    conn.execute("DELETE FROM employees WHERE name = ?", (name,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Attendance – Check-in / Check-out
# ---------------------------------------------------------------------------

def _get_today_record(name: str, conn) -> dict | None:
    """Lấy bản ghi chấm công hôm nay của nhân viên (nếu có)."""
    today = today_vn()
    row = conn.execute(
        "SELECT * FROM attendance WHERE name = ? AND date = ?",
        (name, today),
    ).fetchone()
    return dict(row) if row else None


def record_attendance(name: str) -> dict:
    """
    Logic chấm công thông minh:
    - Lần 1 trong ngày → ghi giờ VÀO (check_in)
    - Lần 2 → ghi giờ RA (check_out) và tính tổng giờ làm
    - Lần 3+ → thông báo đã hoàn tất

    Returns:
        {
          "status": "checked_in" | "checked_out" | "already_done",
          "check_in": str,
          "check_out": str | None,
          "total_hours": float | None,
        }
    """
    conn = get_connection()
    record = _get_today_record(name, conn)
    now = now_vn()
    time_str = now.strftime("%H:%M:%S")

    if record is None:
        # Lần đầu → check-in
        conn.execute(
            "INSERT INTO attendance (name, date, check_in) VALUES (?, ?, ?)",
            (name, today_vn(), time_str),
        )
        conn.commit()
        conn.close()
        return {"status": "checked_in", "check_in": time_str, "check_out": None, "total_hours": None}

    if record["check_out"] is None:
        # Đã check-in nhưng chưa check-out → ghi check-out
        # Tính tổng giờ từ chuỗi HH:MM:SS (tránh vấn đề timezone-aware/naive)
        fmt = "%H:%M:%S"
        t_in  = datetime.strptime(record["check_in"], fmt)
        t_out = datetime.strptime(time_str, fmt)
        # Xử lý trường hợp vượt qua nửa đêm
        delta_seconds = (t_out - t_in).total_seconds()
        if delta_seconds < 0:
            delta_seconds += 86400  # +24h
        total_hours = round(delta_seconds / 3600, 2)

        conn.execute(
            "UPDATE attendance SET check_out = ?, total_hours = ? WHERE id = ?",
            (time_str, total_hours, record["id"]),
        )
        conn.commit()
        conn.close()
        return {
            "status": "checked_out",
            "check_in": record["check_in"],
            "check_out": time_str,
            "total_hours": total_hours,
        }


    # Đã có cả check_in và check_out
    conn.close()
    return {
        "status": "already_done",
        "check_in": record["check_in"],
        "check_out": record["check_out"],
        "total_hours": record["total_hours"],
    }


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def get_attendance_history(filter_date: str | None = None) -> list[dict]:
    """
    Lấy lịch sử chấm công.
    filter_date: 'YYYY-MM-DD', None = lấy tất cả.
    """
    conn = get_connection()
    if filter_date:
        rows = conn.execute(
            "SELECT * FROM attendance WHERE date = ? ORDER BY check_in DESC",
            (filter_date,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM attendance ORDER BY date DESC, check_in DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_attendance_today() -> list[dict]:
    """Lấy danh sách chấm công hôm nay."""
    return get_attendance_history(filter_date=today_vn())


def has_checked_in_today(name: str) -> bool:
    """Kiểm tra nhân viên đã chấm công hôm nay chưa (backward compat)."""
    conn = get_connection()
    row = conn.execute(
        "SELECT id FROM attendance WHERE name = ? AND date = ?",
        (name, today_vn()),
    ).fetchone()
    conn.close()
    return row is not None
