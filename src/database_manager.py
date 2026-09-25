import os
import sqlite3
from datetime import datetime


# ==========================================
# DATABASE PATH
# ==========================================

CURRENT_FILE = os.path.abspath(__file__)

SRC_FOLDER = os.path.dirname(CURRENT_FILE)

PROJECT_ROOT = os.path.dirname(SRC_FOLDER)

DATABASE_FOLDER = os.path.join(
    PROJECT_ROOT,
    "database"
)

DATABASE_PATH = os.path.join(
    DATABASE_FOLDER,
    "attendance.db"
)


# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_connection():

    os.makedirs(
        DATABASE_FOLDER,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


# ==========================================
# INITIALIZE DATABASE
# ==========================================

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()


    # ======================================
    # STUDENTS TABLE
    # ======================================

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS students (

            student_id TEXT PRIMARY KEY,

            name TEXT NOT NULL

        )
        """
    )


    # ======================================
    # ATTENDANCE TABLE
    # ======================================

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id TEXT NOT NULL,

            date TEXT NOT NULL,

            time TEXT NOT NULL,

            status TEXT NOT NULL DEFAULT 'Present',

            FOREIGN KEY (student_id)
                REFERENCES students(student_id),

            UNIQUE(student_id, date)

        )
        """
    )


    connection.commit()

    connection.close()


# ==========================================
# ADD STUDENT
# ==========================================

def add_student(
    student_id,
    name
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        INSERT OR IGNORE INTO students
        (student_id, name)

        VALUES (?, ?)
        """,
        (
            student_id,
            name
        )
    )


    connection.commit()

    connection.close()


# ==========================================
# ADD INITIAL STUDENTS
# ==========================================

def add_initial_students():

    add_student(
        "0201MT241055",
        "Manshu Tiwari"
    )

    add_student(
        "0201AI241033",
        "Shivani Choubey"
    )


# ==========================================
# MARK ATTENDANCE
# ==========================================

def mark_attendance(
    student_id
):

    connection = get_connection()

    cursor = connection.cursor()


    now = datetime.now()

    current_date = now.strftime(
        "%Y-%m-%d"
    )

    current_time = now.strftime(
        "%H:%M:%S"
    )


    cursor.execute(
        """
        INSERT OR IGNORE INTO attendance
        (
            student_id,
            date,
            time,
            status
        )

        VALUES (?, ?, ?, ?)
        """,
        (
            student_id,
            current_date,
            current_time,
            "Present"
        )
    )


    connection.commit()


    # ======================================
    # CHECK WHETHER RECORD WAS INSERTED
    # ======================================

    attendance_marked = (
        cursor.rowcount == 1
    )


    connection.close()


    return attendance_marked


# ==========================================
# GET TODAY'S ATTENDANCE
# ==========================================

def get_today_attendance():

    connection = get_connection()

    cursor = connection.cursor()


    today = datetime.now().strftime(
        "%Y-%m-%d"
    )


    cursor.execute(
        """
        SELECT
            attendance.student_id,
            students.name,
            attendance.date,
            attendance.time,
            attendance.status

        FROM attendance

        INNER JOIN students
            ON attendance.student_id =
               students.student_id

        WHERE attendance.date = ?

        ORDER BY attendance.time
        """,
        (today,)
    )


    records = cursor.fetchall()

    connection.close()


    return records


# ==========================================
# DATABASE TEST
# ==========================================

if __name__ == "__main__":

    print()
    print("========================================")
    print("AI SMART ATTENDANCE DATABASE")
    print("========================================")

    print()
    print("Initializing database...")

    initialize_database()

    add_initial_students()

    print()
    print(
        f"Database created at:"
    )

    print(
        DATABASE_PATH
    )

    print()
    print("Students registered:")

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT student_id, name
        FROM students
        ORDER BY student_id
        """
    )

    students = cursor.fetchall()

    connection.close()


    for student_id, name in students:

        print(
            f"{student_id} -> {name}"
        )


    print()
    print("Database initialization completed.")

    print()
    print("========================================")