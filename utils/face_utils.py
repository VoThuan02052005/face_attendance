"""
face_utils.py - Xử lý nhận diện khuôn mặt bằng InsightFace (ONNX-based)

InsightFace không cần cmake/dlib, chạy hoàn toàn qua ONNX Runtime
→ Phù hợp deploy trên Streamlit Cloud
"""

import os
import numpy as np
import cv2
from PIL import Image

# InsightFace
import insightface
from insightface.app import FaceAnalysis

# Thư mục chứa ảnh nhân viên
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMPLOYEES_DIR = os.path.join(BASE_DIR, "dataset", "employees")

# ── Khởi tạo InsightFace model (singleton) ────────────────────────────────────
_face_app: FaceAnalysis | None = None


def get_face_app() -> FaceAnalysis:
    """
    Trả về singleton FaceAnalysis đã được khởi tạo.
    Dùng model buffalo_sc (nhỏ, nhẹ, phù hợp CPU, không cần GPU).
    """
    global _face_app
    if _face_app is None:
        # buffalo_sc: nhỏ nhất, CPU-friendly, đủ chính xác cho demo
        _face_app = FaceAnalysis(
            name="buffalo_sc",
            root=os.path.join(BASE_DIR, ".insightface"),
            providers=["CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=0, det_size=(320, 320))
    return _face_app


def load_known_faces() -> tuple[list, list]:
    """
    Tải tất cả ảnh từ dataset/employees/ và mã hóa khuôn mặt bằng InsightFace.

    Returns:
        known_embeddings: Danh sách vector embedding (512 chiều với buffalo_sc)
        known_names:      Danh sách tên tương ứng (tên file không có đuôi)
    """
    known_embeddings = []
    known_names = []

    os.makedirs(EMPLOYEES_DIR, exist_ok=True)
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    app = get_face_app()

    for filename in sorted(os.listdir(EMPLOYEES_DIR)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_ext:
            continue

        image_path = os.path.join(EMPLOYEES_DIR, filename)
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise ValueError("Không đọc được file ảnh")

            faces = app.get(img_bgr)
            if faces:
                embedding = faces[0].embedding  # vector 512 chiều
                embedding = embedding / np.linalg.norm(embedding)  # normalize
                known_embeddings.append(embedding)
                name = os.path.splitext(filename)[0]
                known_names.append(name)
        except Exception as e:
            print(f"[WARNING] Không thể load ảnh {filename}: {e}")

    return known_embeddings, known_names


def recognize_faces(
    frame_rgb: np.ndarray,
    known_embeddings: list,
    known_names: list,
    threshold: float = 0.45,
) -> list[dict]:
    """
    Nhận diện khuôn mặt trong một khung hình với InsightFace.

    Args:
        frame_rgb:        Ảnh RGB (numpy array)
        known_embeddings: Danh sách embedding đã biết
        known_names:      Danh sách tên tương ứng
        threshold:        Ngưỡng cosine similarity (≥ threshold → khớp)

    Returns:
        Danh sách dict {"name": str, "location": (x1, y1, x2, y2)}
    """
    results = []

    if not known_embeddings:
        return results

    app = get_face_app()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    faces = app.get(frame_bgr)

    for face in faces:
        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)  # normalize

        # Cosine similarity với tất cả các face đã lưu
        sims = [float(np.dot(embedding, e)) for e in known_embeddings]

        name = "Không nhận diện được"
        best_sim = max(sims) if sims else 0.0

        if best_sim >= threshold:
            best_idx = int(np.argmax(sims))
            name = known_names[best_idx]

        # Bounding box từ InsightFace: (x1, y1, x2, y2)
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        results.append({
            "name": name,
            "location": (x1, y1, x2, y2),   # (left, top, right, bottom)
            "similarity": round(best_sim, 3),
        })

    return results


def draw_results(frame_bgr: np.ndarray, results: list[dict]) -> np.ndarray:
    """
    Vẽ bounding box và tên lên frame BGR.
    """
    for res in results:
        name = res["name"]
        x1, y1, x2, y2 = res["location"]
        sim = res.get("similarity", 0)

        recognized = name != "Không nhận diện được"
        color = (0, 200, 0) if recognized else (0, 0, 220)  # BGR: xanh lá / đỏ

        # Vẽ bounding box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        # Label text
        label = f"{name} ({sim:.0%})" if recognized else name

        # Nền cho nhãn
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.rectangle(frame_bgr, (x1, y2 - th - 10), (x1 + tw + 6, y2), color, cv2.FILLED)

        cv2.putText(
            frame_bgr,
            label,
            (x1 + 3, y2 - 5),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

    return frame_bgr


def save_employee_image(name: str, pil_image: Image.Image) -> str:
    """
    Lưu ảnh nhân viên vào thư mục dataset/employees/.
    """
    os.makedirs(EMPLOYEES_DIR, exist_ok=True)
    safe_name = "_".join(name.strip().split())
    file_path = os.path.join(EMPLOYEES_DIR, f"{safe_name}.jpg")
    pil_image.save(file_path, format="JPEG")
    return file_path


def validate_face_image(pil_image: Image.Image) -> tuple[bool, str]:
    """
    Kiểm tra ảnh có khuôn mặt hợp lệ không bằng InsightFace.

    Returns:
        (True, "") nếu OK
        (False, thông báo lỗi) nếu không hợp lệ
    """
    app = get_face_app()
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = app.get(img_bgr)

    if not faces:
        return False, "Không tìm thấy khuôn mặt trong ảnh. Vui lòng chọn ảnh rõ mặt hơn."
    if len(faces) > 1:
        return False, f"Phát hiện {len(faces)} khuôn mặt. Mỗi ảnh chỉ nên có 1 người."
    return True, ""
