import os
import sys
import cv2
import numpy as np


# ==========================================
# PROJECT PATH
# ==========================================

# evaluate_recognition.py is inside:
# E:\AI-Smart-Attendance\src
#
# Therefore the project root is one level above src.

CURRENT_FILE = os.path.abspath(__file__)
SRC_FOLDER = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(SRC_FOLDER)

# Add src folder to Python import path
if SRC_FOLDER not in sys.path:
    sys.path.insert(0, SRC_FOLDER)


# Enrollment folder:
# E:\AI-Smart-Attendance\data\enrollment

ENROLLMENT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "enrollment"
)


# ==========================================
# IMPORT PROJECT MODULES
# ==========================================

from face_detection import (
    detect_faces,
    preprocess_face
)

from face_embedding import FaceEmbeddingModel

from face_recognition import FaceRecognizer


# ==========================================
# CONFIGURATION
# ==========================================

SAMPLES_PER_PERSON = 20

THRESHOLDS = np.arange(
    0.50,
    0.91,
    0.05
)


STUDENTS = {
    "m": {
        "id": "0201MT241055",
        "name": "Manshu Tiwari"
    },

    "s": {
        "id": "0201AI241033",
        "name": "Shivani Choubey"
    }
}


# ==========================================
# VERIFY ENROLLMENT FOLDER
# ==========================================

print()
print("========================================")
print("AI SMART ATTENDANCE")
print("RECOGNITION EVALUATION")
print("========================================")

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Enrollment folder:")
print(ENROLLMENT_FOLDER)


if not os.path.exists(ENROLLMENT_FOLDER):

    raise FileNotFoundError(
        f"\nEnrollment folder not found:\n"
        f"{ENROLLMENT_FOLDER}"
    )


# ==========================================
# VERIFY EMBEDDING FILES
# ==========================================

required_files = [
    "0201MT241055_embedding.npy",
    "0201AI241033_embedding.npy"
]


for filename in required_files:

    file_path = os.path.join(
        ENROLLMENT_FOLDER,
        filename
    )

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nRequired enrollment file not found:\n"
            f"{file_path}"
        )


print()
print("Enrollment files verified.")


# ==========================================
# LOAD MODELS
# ==========================================

print()
print("========================================")
print("LOADING RECOGNITION MODELS")
print("========================================")


embedding_model = FaceEmbeddingModel()


recognizer = FaceRecognizer(
    enrollment_folder=ENROLLMENT_FOLDER,
    threshold=0.70
)


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


print()
print("Camera started.")


# ==========================================
# SAMPLE STORAGE
# ==========================================

samples = []


# ==========================================
# SAMPLE COLLECTION FUNCTION
# ==========================================

def collect_samples(
    label,
    student_id,
    student_name
):

    print()
    print("========================================")
    print(f"COLLECTING SAMPLES")
    print(f"Person: {student_name}")
    print("========================================")

    print(
        f"Required samples: "
        f"{SAMPLES_PER_PERSON}"
    )

    print()
    print("Press SPACE to capture.")
    print("Press Q to quit.")
    print()

    count = 0

    while count < SAMPLES_PER_PERSON:

        ret, frame = camera.read()

        if not ret:

            print(
                "WARNING: Could not read frame."
            )

            continue


        # ----------------------------------
        # DETECT FACE
        # ----------------------------------

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
            f"{count}/{SAMPLES_PER_PERSON}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )


        # ----------------------------------
        # ONE FACE FOUND
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


            cv2.putText(
                display_frame,
                "SPACE = Capture",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )


        # ----------------------------------
        # ZERO OR MULTIPLE FACES
        # ----------------------------------

        else:

            cv2.putText(
                display_frame,
                "Show exactly ONE face",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )


        # ----------------------------------
        # SHOW CAMERA
        # ----------------------------------

        cv2.imshow(
            "Recognition Evaluation",
            display_frame
        )


        key = cv2.waitKey(1) & 0xFF


        # ----------------------------------
        # QUIT
        # ----------------------------------

        if key == ord("q"):

            camera.release()

            cv2.destroyAllWindows()

            print()
            print(
                "Evaluation cancelled."
            )

            sys.exit(0)


        # ----------------------------------
        # CAPTURE SAMPLE
        # ----------------------------------

        if key == 32:

            if len(faces) != 1:

                print(
                    "Capture rejected: "
                    "exactly one face required."
                )

                continue


            face = faces[0]


            # ----------------------------------
            # PREPROCESS FACE
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


            # ----------------------------------
            # RECOGNITION
            # ----------------------------------

            (
                predicted_id,
                predicted_name,
                similarity,
                _
            ) = recognizer.compare(
                embedding
            )


            # ----------------------------------
            # SAVE RESULT
            # ----------------------------------

            samples.append(
                {
                    "true_label": label,
                    "true_id": student_id,
                    "true_name": student_name,
                    "predicted_id": predicted_id,
                    "predicted_name": predicted_name,
                    "similarity": similarity
                }
            )


            count += 1


            print(
                f"{count:02d}/"
                f"{SAMPLES_PER_PERSON} | "
                f"Similarity: "
                f"{similarity:.4f} | "
                f"Prediction: "
                f"{predicted_name}"
            )


# ==========================================
# ROUND 1
# MANSHU
# ==========================================

collect_samples(
    label="manshu",
    student_id="0201MT241055",
    student_name="Manshu Tiwari"
)


# ==========================================
# ROUND 2
# SHIVANI
# ==========================================

collect_samples(
    label="shivani",
    student_id="0201AI241033",
    student_name="Shivani Choubey"
)


# ==========================================
# ROUND 3
# UNKNOWN PERSON
# ==========================================

collect_samples(
    label="unknown",
    student_id=None,
    student_name="UNKNOWN PERSON"
)


# ==========================================
# RELEASE CAMERA
# ==========================================

camera.release()

cv2.destroyAllWindows()


# ==========================================
# EVALUATION
# ==========================================

print()
print()
print("========================================")
print("RECOGNITION EVALUATION")
print("========================================")


total_samples = len(samples)


print(
    f"Total samples: {total_samples}"
)

print()


# ==========================================
# TEST DIFFERENT THRESHOLDS
# ==========================================

for threshold in THRESHOLDS:

    correct = 0

    false_accepts = 0

    false_rejects = 0


    # --------------------------------------
    # PROCESS SAMPLES
    # --------------------------------------

    for sample in samples:

        similarity = sample[
            "similarity"
        ]

        true_label = sample[
            "true_label"
        ]

        predicted_id = sample[
            "predicted_id"
        ]

        true_id = sample[
            "true_id"
        ]


        # ==================================
        # UNKNOWN PERSON
        # ==================================

        if true_label == "unknown":

            if similarity >= threshold:

                # Unknown person incorrectly
                # accepted.

                false_accepts += 1

            else:

                # Unknown correctly rejected.

                correct += 1


        # ==================================
        # KNOWN PERSON
        # ==================================

        else:

            correct_identity = (
                predicted_id == true_id
            )

            accepted = (
                similarity >= threshold
            )


            if (
                correct_identity
                and accepted
            ):

                correct += 1


            elif not accepted:

                false_rejects += 1


    # --------------------------------------
    # CALCULATE METRICS
    # --------------------------------------

    accuracy = (
        correct /
        total_samples
    ) * 100


    false_accept_rate = (
        false_accepts /
        SAMPLES_PER_PERSON
    ) * 100


    false_reject_rate = (
        false_rejects /
        (SAMPLES_PER_PERSON * 2)
    ) * 100


    # --------------------------------------
    # PRINT RESULT
    # --------------------------------------

    print(
        f"Threshold: {threshold:.2f} | "
        f"Accuracy: "
        f"{accuracy:6.2f}% | "
        f"FAR: "
        f"{false_accept_rate:6.2f}% | "
        f"FRR: "
        f"{false_reject_rate:6.2f}%"
    )


# ==========================================
# FINISHED
# ==========================================

print()
print("========================================")
print("EVALUATION COMPLETE")
print("========================================")