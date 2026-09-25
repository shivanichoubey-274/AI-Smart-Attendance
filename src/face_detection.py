import cv2
import numpy as np


def detect_faces(frame, face_detector):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    return faces


def preprocess_face(frame, face):

    x, y, w, h = face

    # Crop face
    face_image = frame[
        y:y + h,
        x:x + w
    ]

    # Resize
    face_image = cv2.resize(
        face_image,
        (160, 160)
    )

    # BGR → RGB
    face_image = cv2.cvtColor(
        face_image,
        cv2.COLOR_BGR2RGB
    )

    # Convert to float32 and normalize
    face_image = (
        face_image.astype(np.float32) / 255.0
    )

    return face_image


def draw_faces(frame, faces):

    for (x, y, w, h) in faces:

        # Draw face rectangle only
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    return frame