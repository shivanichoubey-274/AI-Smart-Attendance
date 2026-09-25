import os
import numpy as np


class FaceRecognizer:

    def __init__(
        self,
        enrollment_folder,
        threshold=0.70
    ):

        self.enrollment_folder = (
            enrollment_folder
        )

        self.threshold = threshold

        self.students = {
            "0201MT241055": "Manshu Tiwari",
            "0201AI241033": "Shivani Choubey"
        }

        self.embeddings = {}

        self.load_embeddings()


    # ======================================
    # LOAD MULTIPLE EMBEDDINGS
    # ======================================

    def load_embeddings(self):

        for student_id, student_name in (
            self.students.items()
        ):

            embedding_file = os.path.join(
                self.enrollment_folder,
                f"{student_id}_embeddings.npy"
            )

            if not os.path.exists(
                embedding_file
            ):

                raise FileNotFoundError(
                    "Enrollment embeddings not found:\n"
                    f"{embedding_file}"
                )


            embeddings = np.load(
                embedding_file
            ).astype(np.float32)


            # ----------------------------------
            # VERIFY SHAPE
            # ----------------------------------

            if embeddings.ndim != 2:

                raise ValueError(
                    f"Invalid embedding shape for "
                    f"{student_name}: "
                    f"{embeddings.shape}"
                )


            if embeddings.shape[1] != 512:

                raise ValueError(
                    f"Expected 512-dimensional "
                    f"embeddings for "
                    f"{student_name}, got "
                    f"{embeddings.shape[1]}"
                )


            # ----------------------------------
            # NORMALIZE EVERY EMBEDDING
            # ----------------------------------

            normalized_embeddings = []


            for embedding in embeddings:

                norm = np.linalg.norm(
                    embedding
                )

                if norm == 0:
                    continue

                normalized_embeddings.append(
                    embedding / norm
                )


            if len(
                normalized_embeddings
            ) == 0:

                raise ValueError(
                    f"No valid embeddings "
                    f"found for "
                    f"{student_name}"
                )


            self.embeddings[
                student_id
            ] = np.array(
                normalized_embeddings,
                dtype=np.float32
            )


        # ==================================
        # DISPLAY LOADED STUDENTS
        # ==================================

        print(
            "================================"
        )

        print(
            "MULTI-EMBEDDING STUDENTS"
        )

        print(
            "================================"
        )


        for student_id, student_name in (
            self.students.items()
        ):

            count = len(
                self.embeddings[
                    student_id
                ]
            )

            print(
                f"{student_id} -> "
                f"{student_name} "
                f"({count} embeddings)"
            )


        print(
            "================================"
        )


    # ======================================
    # COMPARE LIVE FACE
    # ======================================

    def compare(
        self,
        live_embedding
    ):

        live_embedding = (
            live_embedding
            .astype(np.float32)
        )


        # ----------------------------------
        # NORMALIZE LIVE EMBEDDING
        # ----------------------------------

        norm = np.linalg.norm(
            live_embedding
        )


        if norm == 0:

            return (
                None,
                "UNKNOWN",
                0.0,
                False
            )


        live_embedding = (
            live_embedding / norm
        )


        # ==================================
        # FIND BEST MATCH
        # ==================================

        best_student_id = None

        best_student_name = "UNKNOWN"

        best_similarity = -1.0


        for student_id, embeddings in (
            self.embeddings.items()
        ):

            # ----------------------------------
            # Compare live embedding with all
            # embeddings belonging to student
            # ----------------------------------

            similarities = np.dot(
                embeddings,
                live_embedding
            )


            # Highest similarity for this student

            student_best_similarity = float(
                np.max(similarities)
            )


            # ----------------------------------
            # Keep global best match
            # ----------------------------------

            if (
                student_best_similarity
                > best_similarity
            ):

                best_similarity = (
                    student_best_similarity
                )

                best_student_id = (
                    student_id
                )

                best_student_name = (
                    self.students[
                        student_id
                    ]
                )


        # ==================================
        # APPLY THRESHOLD
        # ==================================

        matched = (
            best_similarity
            >= self.threshold
        )


        if not matched:

            best_student_id = None

            best_student_name = "UNKNOWN"


        return (
            best_student_id,
            best_student_name,
            best_similarity,
            matched
        )