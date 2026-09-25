# AI-Based Smart Attendance System

A real-time AI-powered attendance system that uses computer vision and deep-learning-based face recognition to automatically identify registered students and record their attendance.

## 🚀 Features

- Real-time face detection using OpenCV
- Deep-learning-based face recognition using PyTorch
- 512-dimensional face embeddings
- Multi-sample student enrollment
- Cosine-similarity-based identity matching
- Configurable recognition threshold
- Automatic attendance marking
- Duplicate attendance prevention
- SQLite database for attendance records
- Pandas-based attendance analysis
- Recognition performance evaluation using Accuracy, FAR, and FRR

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Computer Vision:** OpenCV
- **Numerical Computing:** NumPy
- **Data Analysis:** Pandas
- **Database:** SQLite
- **Face Recognition:** InceptionResnetV1 / VGGFace2

## 📂 Project Structure

```text
AI-Smart-Attendance/
│
├── data/
│   └── enrollment/
│
├── database/
│
├── src/
│   ├── camera.py
│   ├── database_manager.py
│   ├── enrollment.py
│   ├── evaluate_recognition.py
│   ├── face_detection.py
│   ├── face_embedding.py
│   ├── face_recognition.py
│   └── multi_enrollment.py
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
└── .gitignore

HOW IT WORK

Camera
   ↓
Face Detection
   ↓
Face Preprocessing
   ↓
Deep Learning Model
   ↓
512-D Face Embedding
   ↓
Cosine Similarity Matching
   ↓
Student Identification
   ↓
SQLite Attendance Database
