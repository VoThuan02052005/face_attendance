"""
db_utils.py - Xử lý cơ sở dữ liệu SQLite cho hệ thống chấm công
"""

import sqlite3
import os
from datetime import datetime, date

# Đường dẫn tuyệt đối tới file database
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

    # Bảng attendance
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            date      TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Employees
# ---------------------------------------------------------------------------

def add_employee(name: str, image_path: str) -> bool:
    """
    Thêm nhân viên mới vào CSDL.
    Trả về True nếu thêm thành công, False nếu tên đã tồn tại.
    """
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
# Attendance
# ---------------------------------------------------------------------------

def has_checked_in_today(name: str) -> bool:
    """
    Kiểm tra nhân viên đã chấm công hôm nay chưa.
    Tránh ghi trùng nhiều lần trong cùng một ngày.
    """
    today = date.today().isoformat()
    conn = get_connection()
    row = conn.execute(
        "SELECT id FROM attendance WHERE name = ? AND date = ?",
        (name, today),
    ).fetchone()
    conn.close()
    return row is not None


def record_attendance(name: str) -> bool:
    """
    Ghi chấm công cho nhân viên.
    Trả về True nếu ghi thành công, False nếu đã chấm công hôm nay.
    """
    if has_checked_in_today(name):
        return False

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    today = now.date().isoformat()

    conn = get_connection()
    conn.execute(
        "INSERT INTO attendance (name, timestamp, date) VALUES (?, ?, ?)",
        (name, timestamp, today),
    )
    conn.commit()
    conn.close()
    return True


def get_attendance_history(filter_date: str | None = None) -> list[dict]:
    """
    Lấy lịch sử chấm công.
    filter_date: chuỗi 'YYYY-MM-DD', nếu None thì lấy tất cả.
    """
    conn = get_connection()
    if filter_date:
        rows = conn.execute(
            "SELECT * FROM attendance WHERE date = ? ORDER BY timestamp DESC",
            (filter_date,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM attendance ORDER BY timestamp DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_attendance_today() -> list[dict]:
    """Lấy danh sách chấm công hôm nay."""
    return get_attendance_history(filter_date=date.today().isoformat())
