import cv2


def detect_fog(frame):
    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate image contrast
    contrast = gray.std()

    # Calculate brightness
    brightness = gray.mean()

    # Simple fog detection threshold
    if contrast < 35:
        return True, contrast, "FOG / LOW VISIBILITY"

    return False, contrast, "NORMAL VISIBILITY"


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera could not be opened")
    exit()

print("Fog detection started...")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame")
        break

    fog_detected, contrast, status = detect_fog(frame)

    # --------------------------------
    # STATUS DISPLAY
    # --------------------------------

    status_color = (0, 0, 255) if fog_detected else (0, 255, 0)

    # Black background behind status
    cv2.rectangle(
        frame,
        (10, 10),
        (390, 65),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        status,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2
    )

    # --------------------------------
    # CONTRAST DISPLAY
    # --------------------------------

    cv2.rectangle(
        frame,
        (10, 75),
        (250, 115),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Contrast: {contrast:.2f}",
        (20, 103),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------
    # SHOW CAMERA
    # --------------------------------

    cv2.imshow("AI Fog Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()