# Hướng dẫn triển khai hệ thống chấm công bằng nhận diện khuôn mặt

## 1. Giới thiệu

Dự án này xây dựng một **hệ thống chấm công tự động bằng nhận diện khuôn mặt** sử dụng **Python, OpenCV và Streamlit**.

Hệ thống cho phép:

* Đăng ký nhân viên bằng ảnh khuôn mặt
* Nhận diện khuôn mặt thông qua webcam
* Tự động ghi lại thời gian chấm công và có hiển thị thông báo chấm công.
* Hiển thị lịch sử chấm công


Mục tiêu của dự án là minh họa cách tích hợp **Computer Vision vào một ứng dụng web đơn giản**.

---

# 2. Kiến trúc hệ thống

Hệ thống hoạt động theo pipeline sau:

Camera → Phát hiện khuôn mặt → Mã hóa khuôn mặt → So sánh dữ liệu → Lưu chấm công

### Các thành phần chính

* **Giao diện web:** Streamlit
* **Xử lý ảnh:** OpenCV
* **Nhận diện khuôn mặt:** thư viện face_recognition
* **Cơ sở dữ liệu:** SQLite
* **Xử lý dữ liệu:** NumPy, Pandas

---

# 3. Công nghệ sử dụng

| Thành phần          | Công nghệ        |
| ------------------- | ---------------- |
| Giao diện web       | Streamlit        |
| Backend             | Python           |
| Computer Vision     | OpenCV           |
| Nhận diện khuôn mặt | face_recognition |
| Cơ sở dữ liệu       | SQLite           |
| Xử lý dữ liệu       | NumPy, Pandas    |

---

# 4. Cấu trúc thư mục dự án

```text
face_attendance_system
│
├── app.py
│
├── database
│   └── attendance.db
│
├── dataset
│   └── employees
│       ├── employee1.jpg
│       ├── employee2.jpg
│
├── utils
│   ├── face_utils.py
│   └── db_utils.py
│
└── README.md
```

---

# 5. Quy trình hoạt động của hệ thống

## 5.1 Đăng ký nhân viên

1. Người dùng tải ảnh nhân viên lên từ giao diện Streamlit
2. Ảnh được lưu vào thư mục dataset
3. Hệ thống tạo vector đặc trưng khuôn mặt (face encoding)
4. Lưu thông tin nhân viên

---

## 5.2 Quy trình chấm công

1. Camera webcam ghi hình
2. Hệ thống phát hiện khuôn mặt trong khung hình
3. Chuyển khuôn mặt thành vector đặc trưng
4. So sánh với dữ liệu khuôn mặt đã lưu
5. Nếu khớp:

   * Xác định tên nhân viên
   * Ghi thời gian chấm công
   * Lưu vào cơ sở dữ liệu

---

# 6. Pipeline nhận diện khuôn mặt

Thư viện **face_recognition** sử dụng mô hình học sâu từ **dlib**.

Các bước xử lý:

### Bước 1: Phát hiện khuôn mặt

Xác định vị trí khuôn mặt trong ảnh.

### Bước 2: Mã hóa khuôn mặt

Chuyển khuôn mặt thành **vector 128 chiều**.

### Bước 3: So sánh khuôn mặt

So sánh vector với dữ liệu đã lưu để tìm người trùng khớp.

---

# 7. Thiết kế cơ sở dữ liệu

## Bảng employees

| Cột        | Kiểu dữ liệu |
| ---------- | ------------ |
| id         | INTEGER      |
| name       | TEXT         |
| image_path | TEXT         |

---

## Bảng attendance

| Cột       | Kiểu dữ liệu |
| --------- | ------------ |
| id        | INTEGER      |
| name      | TEXT         |
| timestamp | TEXT         |

---

# 8. Cài đặt môi trường

Cài đặt các thư viện cần thiết:

```bash
pip install streamlit
pip install opencv-python
pip install face-recognition
pip install numpy
pip install pandas
```

---

# 9. Chạy ứng dụng

Khởi động ứng dụng bằng lệnh:

```bash
streamlit run app.py
```

Sau khi chạy, trình duyệt sẽ mở tại địa chỉ:

```
http://localhost:8501
```

---

# 10. Các chức năng chính

Hệ thống hỗ trợ các chức năng:

* Đăng ký nhân viên bằng ảnh
* Nhận diện khuôn mặt qua webcam
* Ghi lại thời gian chấm công tự động
* Xem lịch sử chấm công
* Giao diện web đơn giản dễ sử dụng

---

# 11. Hướng phát triển trong tương lai

Hệ thống có thể được mở rộng với các tính năng:

* Phát hiện người thật (Liveness Detection)
* Hỗ trợ nhiều camera
* Kết nối cơ sở dữ liệu cloud
* Dashboard thống kê chấm công
* Quản lý nhân viên

---

# 12. Kết luận

Dự án minh họa cách xây dựng một hệ thống **chấm công tự động bằng nhận diện khuôn mặt** kết hợp giữa **Computer Vision và ứng dụng web**.

Hệ thống nhẹ, dễ triển khai và phù hợp cho:

* học tập
* nghiên cứu
* demo dự án
* prototype hệ thống AI.
