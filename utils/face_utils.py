"""
face_utils.py - Xử lý nhận diện khuôn mặt cho hệ thống chấm công
"""

import os
import face_recognition
import numpy as np
from PIL import Image
import cv2

# Thư mục chứa ảnh nhân viên
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMPLOYEES_DIR = os.path.join(BASE_DIR, "dataset", "employees")


def load_known_faces() -> tuple[list, list]:
    """
    Tải tất cả ảnh từ thư mục dataset/employees/ và mã hóa khuôn mặt.

    Returns:
        known_encodings: Danh sách vector 128 chiều (face encoding)
        known_names:     Danh sách tên tương ứng (tên file không có đuôi)
    """
    known_encodings = []
    known_names = []

    os.makedirs(EMPLOYEES_DIR, exist_ok=True)
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for filename in os.listdir(EMPLOYEES_DIR):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_ext:
            continue

        image_path = os.path.join(EMPLOYEES_DIR, filename)
        try:
            img = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_encodings.append(encodings[0])
                # Tên nhân viên lấy từ tên file (bỏ phần mở rộng)
                name = os.path.splitext(filename)[0]
                known_names.append(name)
        except Exception as e:
            print(f"[WARNING] Không thể load ảnh {filename}: {e}")

    return known_encodings, known_names


def recognize_faces(
    frame_rgb: np.ndarray,
    known_encodings: list,
    known_names: list,
    tolerance: float = 0.5,
) -> list[dict]:
    """
    Nhận diện khuôn mặt trong một khung hình.

    Args:
        frame_rgb:       Ảnh RGB (numpy array)
        known_encodings: Danh sách encoding đã biết
        known_names:     Danh sách tên tương ứng
        tolerance:       Ngưỡng nhận diện (càng nhỏ càng nghiêm)

    Returns:
        Danh sách dict {"name": str, "location": (top, right, bottom, left)}
    """
    results = []

    if not known_encodings:
        return results

    # Scale down để tăng tốc xử lý
    small_frame = cv2.resize(frame_rgb, (0, 0), fx=0.5, fy=0.5)

    face_locations = face_recognition.face_locations(small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for encoding, location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=tolerance)
        face_distances = face_recognition.face_distance(known_encodings, encoding)

        name = "Không nhận diện được"
        if any(matches):
            best_idx = int(np.argmin(face_distances))
            if matches[best_idx]:
                name = known_names[best_idx]

        # Scale vị trí về kích thước gốc
        top, right, bottom, left = location
        top    *= 2
        right  *= 2
        bottom *= 2
        left   *= 2

        results.append({"name": name, "location": (top, right, bottom, left)})

    return results


def draw_results(frame_bgr: np.ndarray, results: list[dict]) -> np.ndarray:
    """
    Vẽ bounding box và tên lên frame BGR (để hiển thị với OpenCV / chuyển về PIL).

    Args:
        frame_bgr: Ảnh BGR (numpy array)
        results:   Kết quả từ recognize_faces()

    Returns:
        Ảnh BGR đã được vẽ bounding box
    """
    for res in results:
        name = res["name"]
        top, right, bottom, left = res["location"]

        recognized = name != "Không nhận diện được"
        color = (0, 200, 0) if recognized else (0, 0, 220)  # BGR: xanh lá / đỏ

        # Vẽ khung
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)

        # Nền cho nhãn tên
        cv2.rectangle(frame_bgr, (left, bottom - 28), (right, bottom), color, cv2.FILLED)

        # Tên nhân viên
        cv2.putText(
            frame_bgr,
            name,
            (left + 4, bottom - 8),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

    return frame_bgr


def save_employee_image(name: str, pil_image: Image.Image) -> str:
    """
    Lưu ảnh nhân viên vào thư mục dataset/employees/.

    Args:
        name:      Tên nhân viên (dùng làm tên file)
        pil_image: Ảnh PIL

    Returns:
        Đường dẫn tuyệt đối tới file đã lưu
    """
    os.makedirs(EMPLOYEES_DIR, exist_ok=True)
    # Chuẩn hóa tên file: bỏ khoảng trắng, ký tự đặc biệt
    safe_name = "_".join(name.strip().split())
    file_path = os.path.join(EMPLOYEES_DIR, f"{safe_name}.jpg")
    pil_image.save(file_path, format="JPEG")
    return file_path


def validate_face_image(pil_image: Image.Image) -> tuple[bool, str]:
    """
    Kiểm tra ảnh có khuôn mặt hợp lệ không.

    Returns:
        (True, "") nếu OK
        (False, thông báo lỗi) nếu không hợp lệ
    """
    img_array = np.array(pil_image.convert("RGB"))
    encodings = face_recognition.face_encodings(img_array)
    if not encodings:
        return False, "Không tìm thấy khuôn mặt trong ảnh. Vui lòng chọn ảnh rõ mặt hơn."
    if len(encodings) > 1:
        return False, f"Phát hiện {len(encodings)} khuôn mặt. Mỗi ảnh chỉ nên có 1 người."
    return True, ""
