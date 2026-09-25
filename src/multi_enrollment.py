import os
import sys
import cv2
import numpy as np


# ==========================================
# PROJECT PATH
# ==========================================

CURRENT_FILE = os.path.abspath(__file__)
SRC_FOLDER = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(SRC_FOLDER)

ENROLLMENT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "enrollment"
)


# ==========================================
# IMPORT MODULES
# ==========================================

if SRC_FOLDER not in sys.path:
    sys.path.insert(0, SRC_FOLDER)

from face_detection import detect_faces, preprocess_face
from face_embedding import FaceEmbeddingModel


# ==========================================
# CONFIGURATION
# ==========================================

SAMPLES_PER_STUDENT = 10

STUDENTS = {
    "1": {
        "id": "0201MT241055",
        "name": "Manshu Tiwari"
    },
    "2": {
        "id": "0201AI241033",
        "name": "Shivani Choubey"
    }
}


# ==========================================
# MOUSE CONTROL
# ==========================================

capture_requested = False
quit_requested = False


def mouse_callback(event, x, y, flags, param):

    global capture_requested
    global quit_requested

    # LEFT MOUSE BUTTON = CAPTURE
    if event == cv2.EVENT_LBUTTONDOWN:
        capture_requested = True

    # RIGHT MOUSE BUTTON = QUIT
    elif event == cv2.EVENT_RBUTTONDOWN:
        quit_requested = True


# ==========================================
# PREPARE FOLDER
# ==========================================

os.makedirs(
    ENROLLMENT_FOLDER,
    exist_ok=True
)


# ==========================================
# LOAD MODELS
# ==========================================

print()
print("========================================")
print("MULTI-EMBEDDING ENROLLMENT")
print("========================================")

print()
print("Loading face embedding model...")

embedding_model = FaceEmbeddingModel()

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    raise RuntimeError(
        "Haar Cascade could not be loaded."
    )

print("Models loaded successfully.")


# ==========================================
# CAMERA SETUP
# ==========================================

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

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

if not camera.isOpened():
    raise RuntimeError(
        "Camera could not be opened."
    )


# ==========================================
# ENROLLMENT
# ==========================================

def enroll_student(student_id, student_name):

    global capture_requested
    global quit_requested

    print()
    print("========================================")
    print("STUDENT ENROLLMENT")
    print("========================================")

    print(f"Student : {student_name}")
    print(f"Roll No : {student_id}")
    print(f"Samples : {SAMPLES_PER_STUDENT}")

    print()
    print("LEFT CLICK  = Capture")
    print("RIGHT CLICK = Quit")
    print()

    embeddings = []

    count = 0

    window_name = "Multi-Embedding Enrollment"

    cv2.namedWindow(window_name)

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    while count < SAMPLES_PER_STUDENT:

        ret, frame = camera.read()

        if not ret:
            print(
                "WARNING: Camera frame could not be read."
            )
            continue

        faces = detect_faces(
            frame,
            face_detector
        )

        display_frame = frame.copy()

        # ----------------------------------
        # HEADER
        # ----------------------------------

        cv2.putText(
            display_frame,
            f"{student_name} | "
            f"{count}/{SAMPLES_PER_STUDENT}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        # ----------------------------------
        # INSTRUCTIONS
        # ----------------------------------

        cv2.putText(
            display_frame,
            "LEFT CLICK = Capture",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            "RIGHT CLICK = Quit",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ----------------------------------
        # FACE DETECTION
        # ----------------------------------

        if len(faces) == 1:

            face = faces[0]

            x, y, w, h = face

            cv2.rectangle(
                display_frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                display_frame,
                "Show exactly ONE face",
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 0, 255),
                2
            )

        # ----------------------------------
        # DISPLAY
        # ----------------------------------

        cv2.imshow(
            window_name,
            display_frame
        )

        # Process window events
        cv2.waitKey(1)

        # ----------------------------------
        # QUIT
        # ----------------------------------

        if quit_requested:

            camera.release()
            cv2.destroyAllWindows()

            print()
            print("Enrollment cancelled.")

            sys.exit(0)

        # ----------------------------------
        # CAPTURE REQUEST
        # ----------------------------------

        if capture_requested:

            # Reset immediately
            capture_requested = False

            # Require exactly one face
            if len(faces) != 1:

                print(
                    "Capture rejected: "
                    "exactly one face required."
                )

                continue

            face = faces[0]

            # ----------------------------------
            # PREPROCESS
            # ----------------------------------

            face_image = preprocess_face(
                frame,
                face
            )

            # ----------------------------------
            # GENERATE EMBEDDING
            # ----------------------------------

            embedding = (
                embedding_model
                .generate_embedding(
                    face_image
                )
            )

            embedding = embedding.astype(
                np.float32
            )

            # ----------------------------------
            # NORMALIZE
            # ----------------------------------

            norm = np.linalg.norm(
                embedding
            )

            if norm == 0:

                print(
                    "Invalid embedding. "
                    "Capture rejected."
                )

                continue

            embedding = embedding / norm

            # ----------------------------------
            # STORE
            # ----------------------------------

            embeddings.append(
                embedding
            )

            count += 1

            print(
                f"Captured "
                f"{count:02d}/"
                f"{SAMPLES_PER_STUDENT}"
            )

    # ==========================================
    # SAVE EMBEDDINGS
    # ==========================================

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    output_file = os.path.join(
        ENROLLMENT_FOLDER,
        f"{student_id}_embeddings.npy"
    )

    np.save(
        output_file,
        embeddings
    )

    print()
    print("========================================")
    print("ENROLLMENT SUCCESSFUL")
    print("========================================")

    print(
        f"Student: {student_name}"
    )

    print(
        f"Roll No: {student_id}"
    )

    print(
        f"Embeddings shape: "
        f"{embeddings.shape}"
    )

    print(
        "Saved to:"
    )

    print(output_file)

    print("========================================")


# ==========================================
# STUDENT SELECTION
# ==========================================

print()
print("Select student:")

print(
    "1. Manshu Tiwari (0201MT241055)"
)

print(
    "2. Shivani Choubey (0201AI241033)"
)

choice = input(
    "\nEnter choice (1/2): "
).strip()


if choice == "1":

    student_id = "0201MT241055"
    student_name = "Manshu Tiwari"

elif choice == "2":

    student_id = "0201AI241033"
    student_name = "Shivani Choubey"

else:

    raise ValueError(
        "Invalid student selection."
    )


# ==========================================
# RUN
# ==========================================

try:

    enroll_student(
        student_id,
        student_name
    )

finally:

    camera.release()
    cv2.destroyAllWindows()