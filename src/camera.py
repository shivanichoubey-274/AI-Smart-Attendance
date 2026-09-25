import cv2


def start_camera():

    print("Opening camera...")

    # Force Windows DirectShow backend
    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the camera."
        )

    # Force MJPG camera format
    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    # Set resolution
    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    # Set FPS
    camera.set(
        cv2.CAP_PROP_FPS,
        30
    )

    print("Camera opened.")
    print("Press Q to exit.")

    while True:

        success, frame = camera.read()

        if not success or frame is None:
            print("Failed to read frame from camera.")
            continue

        cv2.imshow(
            "AI Smart Attendance - Camera Test",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()