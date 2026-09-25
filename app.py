import cv2
import time

from src.face_detection import (
    detect_faces,
    preprocess_face
)

from src.face_embedding import FaceEmbeddingModel

from src.face_recognition import FaceRecognizer

from src.database_manager import (
    initialize_database,
    add_initial_students,
    mark_attendance
)


# ==========================================
# CONFIGURATION
# ==========================================

ENROLLMENT_FOLDER = "data/enrollment"

THRESHOLD = 0.70


# ==========================================
# INITIALIZE DATABASE
# ==========================================

initialize_database()

add_initial_students()


# ==========================================
# LOAD MODELS
# ==========================================

print()
print("========================================")
print("AI SMART ATTENDANCE")
print("========================================")

print()
print("Loading models...")


embedding_model = FaceEmbeddingModel()


recognizer = FaceRecognizer(
    enrollment_folder=ENROLLMENT_FOLDER,
    threshold=THRESHOLD
)


face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


if face_detector.empty():

    raise RuntimeError(
        "Haar Cascade could not be loaded."
    )


print()
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


print()
print("Camera started.")

print()
print("========================================")
print("LIVE RECOGNITION + ATTENDANCE")
print("========================================")
print("Press Q to quit.")
print()


# ==========================================
# ATTENDANCE DISPLAY STATE
# ==========================================

attendance_message = ""

attendance_message_time = 0

MESSAGE_DURATION = 3


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = camera.read()


    if not ret:

        print(
            "ERROR: Could not read camera frame."
        )

        break


    # ======================================
    # FACE DETECTION
    # ======================================

    faces = detect_faces(
        frame,
        face_detector
    )


    # ======================================
    # PROCESS EACH FACE
    # ======================================

    for face in faces:

        x, y, w, h = face


        # ==================================
        # PREPROCESS FACE
        # ==================================

        face_image = preprocess_face(
            frame,
            face
        )


        # ==================================
        # GENERATE EMBEDDING
        # ==================================

        live_embedding = (
            embedding_model
            .generate_embedding(
                face_image
            )
        )


        # ==================================
        # RECOGNIZE FACE
        # ==================================

        (
            student_id,
            student_name,
            similarity,
            matched
        ) = recognizer.compare(
            live_embedding
        )


        # ==================================
        # MATCHED STUDENT
        # ==================================

        if matched:

            # ----------------------------------
            # MARK ATTENDANCE
            # ----------------------------------

            attendance_marked = (
                mark_attendance(
                    student_id
                )
            )


            # ----------------------------------
            # DISPLAY ATTENDANCE STATUS
            # ----------------------------------

            if attendance_marked:

                attendance_message = (
                    f"ATTENDANCE MARKED: "
                    f"{student_name}"
                )

                attendance_message_time = (
                    time.time()
                )

                print(
                    f"[ATTENDANCE] "
                    f"{student_id} | "
                    f"{student_name} | "
                    f"Present"
                )

            else:

                attendance_message = (
                    f"ALREADY MARKED: "
                    f"{student_name}"
                )

                attendance_message_time = (
                    time.time()
                )


            # ----------------------------------
            # FACE LABEL
            # ----------------------------------

            label = (
                f"{student_name} | "
                f"{student_id}"
            )

            score_label = (
                f"Similarity: "
                f"{similarity:.3f}"
            )

            color = (
                0,
                255,
                0
            )


        # ==================================
        # UNKNOWN FACE
        # ==================================

        else:

            label = "UNKNOWN"

            score_label = (
                f"Similarity: "
                f"{similarity:.3f}"
            )

            color = (
                0,
                0,
                255
            )


        # ==================================
        # DRAW FACE BOX
        # ==================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )


        # ==================================
        # STUDENT NAME / UNKNOWN
        # ==================================

        cv2.putText(
            frame,
            label,
            (x, y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )


        # ==================================
        # SIMILARITY
        # ==================================

        cv2.putText(
            frame,
            score_label,
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2
        )


    # ======================================
    # ATTENDANCE STATUS MESSAGE
    # ======================================

    if (
        time.time()
        - attendance_message_time
        < MESSAGE_DURATION
    ):

        cv2.putText(
            frame,
            attendance_message,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )


    # ======================================
    # CAMERA WINDOW
    # ======================================

    cv2.imshow(
        "AI Smart Attendance",
        frame
    )


    # ======================================
    # KEYBOARD
    # ======================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

cv2.destroyAllWindows()


print()
print("Camera stopped.")
print("Program terminated.")