import os
import cv2
import numpy as np

from src.face_detection import detect_faces, preprocess_face
from src.face_embedding import FaceEmbeddingModel


# ==========================================
# STUDENT INFORMATION
# ==========================================

STUDENT_NAME = "Shivani Choubey"
STUDENT_ID = "0201AI241033"


# ==========================================
# FILE LOCATION
# ==========================================

ENROLLMENT_DIR = "data/enrollment"

EMBEDDING_FILE = os.path.join(
    ENROLLMENT_DIR,
    f"{STUDENT_ID}_embedding.npy"
)


def main():

    # Create enrollment directory
    os.makedirs(
        ENROLLMENT_DIR,
        exist_ok=True
    )

    # ======================================
    # LOAD FACE DETECTOR
    # ======================================

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    if face_detector.empty():
        raise RuntimeError(
            "Could not load face detection model."
        )

    # ======================================
    # LOAD FACE EMBEDDING MODEL
    # ======================================

    embedding_model = FaceEmbeddingModel()

    print()
    print("================================")
    print("FACE ENROLLMENT")
    print("================================")
    print(f"Student: {STUDENT_NAME}")
    print(f"Roll No: {STUDENT_ID}")
    print()
    print("Only Shivani should be visible.")
    print("Look at the camera.")
    print()
    print("Press E to enroll.")
    print("Press Q to quit.")
    print()

    # ======================================
    # OPEN CAMERA
    # ======================================

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open camera."
        )

    # Use same working camera settings
    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        30
    )

    # ======================================
    # CAMERA WARM-UP
    # ======================================

    for _ in range(30):

        success, frame = camera.read()

        if success and frame is not None:
            break

    # ======================================
    # CAMERA LOOP
    # ======================================

    while True:

        success, frame = camera.read()

        if not success:
            print(
                "Could not read camera frame."
            )
            continue

        # Detect faces
        faces = detect_faces(
            frame,
            face_detector
        )

        # Draw face boxes
        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Display instructions
        cv2.putText(
            frame,
            "E = Enroll | Q = Quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AI Smart Attendance - Enrollment",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # ==================================
        # ENROLL
        # ==================================

        if key == ord("e"):

            if len(faces) == 0:

                print(
                    "No face detected. "
                    "Try again."
                )

                continue

            if len(faces) > 1:

                print(
                    "Multiple faces detected. "
                    "Only Shivani should be visible."
                )

                continue

            # Get detected face
            face = faces[0]

            # Preprocess
            processed_face = preprocess_face(
                frame,
                face
            )

            # Generate 512-D embedding
            embedding = (
                embedding_model.generate_embedding(
                    processed_face
                )
            )

            # Save embedding
            np.save(
                EMBEDDING_FILE,
                embedding
            )

            print()
            print("================================")
            print("ENROLLMENT SUCCESSFUL")
            print("================================")
            print(f"Student: {STUDENT_NAME}")
            print(f"Roll No: {STUDENT_ID}")
            print(
                f"Embedding shape: "
                f"{embedding.shape}"
            )
            print(
                f"Saved to: "
                f"{EMBEDDING_FILE}"
            )
            print()

            break

        # ==================================
        # QUIT
        # ==================================

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()