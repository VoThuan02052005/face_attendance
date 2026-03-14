"""
db_utils.py - Hệ thống chấm công đa phiên (Multi-Session)
- Quét lẻ (1, 3, 5…) → Giờ VÀO
- Quét chẵn (2, 4, 6…) → Giờ RA
- Báo cáo tổng giờ cuối tuần (Thứ 2–Chủ nhật)
Múi giờ: Việt Nam (UTC+7)
"""

import sqlite3
import os
from datetime import datetime, timezone, timedelta, date

# ── Múi giờ Việt Nam ──────────────────────────────────────────────────────────
VN_TZ = timezone(timedelta(hours=7))


def now_vn() -> datetime:
    """Thời gian hiện tại theo giờ Việt Nam."""
    return datetime.now(VN_TZ)


def today_vn() -> str:
    """Ngày hôm nay theo giờ VN (YYYY-MM-DD)."""
    return now_vn().strftime("%Y-%m-%d")


def current_week_range() -> tuple[str, str]:
    """
    Trả về (ngày_đầu_tuần, ngày_cuối_tuần) theo giờ VN.
    Tuần bắt đầu từ Thứ Hai, kết thúc Chủ Nhật.
    """
    today = now_vn().date()
    monday = today - timedelta(days=today.weekday())       # weekday() = 0 là Thứ Hai
    sunday = monday + timedelta(days=6)
    return monday.isoformat(), sunday.isoformat()


# ── Database ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "attendance.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Tạo các bảng cần thiết nếu chưa tồn tại."""
    conn = get_connection()
    c = conn.cursor()

    # Bảng nhân viên
    c.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL UNIQUE,
            image_path TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)

    # Bảng phiên chấm công (mỗi cặp vào-ra = 1 phiên)
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance_sessions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT NOT NULL,
            date           TEXT NOT NULL,
            session_num    INTEGER NOT NULL DEFAULT 1,
            check_in       TEXT,
            check_out      TEXT,
            duration_hours REAL
        )
    """)

    conn.commit()
    conn.close()


# ── Employees ─────────────────────────────────────────────────────────────────

def add_employee(name: str, image_path: str) -> bool:
    try:
        conn = get_connection()
        conn.execute("INSERT INTO employees (name, image_path) VALUES (?, ?)", (name, image_path))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def get_all_employees() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM employees ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_employee(name: str):
    conn = get_connection()
    conn.execute("DELETE FROM employees WHERE name = ?", (name,))
    conn.commit()
    conn.close()


# ── Attendance – Multi-session Logic ──────────────────────────────────────────

def _get_open_session(name: str, today: str, conn) -> dict | None:
    """Lấy phiên mở (đã check_in nhưng chưa check_out) của hôm nay."""
    row = conn.execute(
        """SELECT * FROM attendance_sessions
           WHERE name = ? AND date = ? AND check_out IS NULL
           ORDER BY session_num DESC LIMIT 1""",
        (name, today),
    ).fetchone()
    return dict(row) if row else None


def _get_next_session_num(name: str, today: str, conn) -> int:
    """Số phiên tiếp theo trong ngày (đếm tất cả phiên hiện có + 1)."""
    row = conn.execute(
        "SELECT COUNT(*) FROM attendance_sessions WHERE name = ? AND date = ?",
        (name, today),
    ).fetchone()
    return (row[0] if row else 0) + 1


def record_attendance(name: str) -> dict:
    """
    Logic chấm công đa phiên (luân phiên):
    - Nếu không có phiên mở → tạo phiên mới, ghi giờ VÀO (check_in)
    - Nếu có phiên mở → ghi giờ RA (check_out), tính duration

    Returns dict:
        status       : "checked_in" | "checked_out"
        session_num  : int
        check_in     : str (HH:MM:SS)
        check_out    : str | None
        duration_hours: float | None
        date         : str (YYYY-MM-DD)
    """
    conn = get_connection()
    today = today_vn()
    time_str = now_vn().strftime("%H:%M:%S")

    open_session = _get_open_session(name, today, conn)

    if open_session is None:
        # ── Quét lẻ → GHI GIỜ VÀO ───────────────────────────────────────────
        session_num = _get_next_session_num(name, today, conn)
        conn.execute(
            """INSERT INTO attendance_sessions (name, date, session_num, check_in)
               VALUES (?, ?, ?, ?)""",
            (name, today, session_num, time_str),
        )
        conn.commit()
        conn.close()
        return {
            "status": "checked_in",
            "session_num": session_num,
            "check_in": time_str,
            "check_out": None,
            "duration_hours": None,
            "date": today,
        }
    else:
        # ── Quét chẵn → GHI GIỜ RA + TÍNH TỔNG ─────────────────────────────
        fmt = "%H:%M:%S"
        t_in  = datetime.strptime(open_session["check_in"], fmt)
        t_out = datetime.strptime(time_str, fmt)
        delta = (t_out - t_in).total_seconds()
        if delta < 0:
            delta += 86400  # vượt qua nửa đêm
        duration_hours = round(delta / 3600, 2)

        conn.execute(
            """UPDATE attendance_sessions
               SET check_out = ?, duration_hours = ?
               WHERE id = ?""",
            (time_str, duration_hours, open_session["id"]),
        )
        conn.commit()
        conn.close()
        return {
            "status": "checked_out",
            "session_num": open_session["session_num"],
            "check_in": open_session["check_in"],
            "check_out": time_str,
            "duration_hours": duration_hours,
            "date": today,
        }


# ── Queries ───────────────────────────────────────────────────────────────────

def get_sessions_today() -> list[dict]:
    """Lấy tất cả phiên chấm công hôm nay."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT * FROM attendance_sessions
           WHERE date = ? ORDER BY name, session_num""",
        (today_vn(),),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_sessions_by_date(filter_date: str | None = None) -> list[dict]:
    """Lấy tất cả phiên theo ngày (None = toàn bộ lịch sử)."""
    conn = get_connection()
    if filter_date:
        rows = conn.execute(
            "SELECT * FROM attendance_sessions WHERE date = ? ORDER BY name, session_num",
            (filter_date,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM attendance_sessions ORDER BY date DESC, name, session_num"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_weekly_summary() -> list[dict]:
    """
    Tổng hợp số giờ làm việc trong tuần hiện tại (Thứ 2 → Chủ nhật).
    Trả về: [{name, total_hours, sessions_count, days_worked}]
    """
    monday, sunday = current_week_range()
    conn = get_connection()
    rows = conn.execute(
        """SELECT
               name,
               ROUND(SUM(COALESCE(duration_hours, 0)), 2)  AS total_hours,
               COUNT(*)                                     AS sessions_count,
               COUNT(DISTINCT date)                         AS days_worked
           FROM attendance_sessions
           WHERE date BETWEEN ? AND ?
             AND check_out IS NOT NULL
           GROUP BY name
           ORDER BY total_hours DESC""",
        (monday, sunday),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_daily_summary_this_week() -> list[dict]:
    """
    Chi tiết từng ngày trong tuần, tổng giờ mỗi người mỗi ngày.
    Trả về: [{name, date, total_hours, sessions_count}]
    """
    monday, sunday = current_week_range()
    conn = get_connection()
    rows = conn.execute(
        """SELECT
               name,
               date,
               ROUND(SUM(COALESCE(duration_hours, 0)), 2) AS total_hours,
               COUNT(*) AS sessions_count
           FROM attendance_sessions
           WHERE date BETWEEN ? AND ?
             AND check_out IS NOT NULL
           GROUP BY name, date
           ORDER BY date, name""",
        (monday, sunday),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Compat helpers ────────────────────────────────────────────────────────────

def get_attendance_today() -> list[dict]:
    """Alias cho get_sessions_today() để giữ compat với app.py."""
    return get_sessions_today()


def get_attendance_history(filter_date: str | None = None) -> list[dict]:
    return get_sessions_by_date(filter_date)


def has_checked_in_today(name: str) -> bool:
    conn = get_connection()
    row = conn.execute(
        "SELECT id FROM attendance_sessions WHERE name = ? AND date = ?",
        (name, today_vn()),
    ).fetchone()
    conn.close()
    return row is not None
