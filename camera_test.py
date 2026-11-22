import cv2

print("1. Starting...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("2. Camera opened:", cap.isOpened())

if not cap.isOpened():
    print("Camera cannot be opened")
    input("Press Enter...")
    exit()

ret, frame = cap.read()

print("3. Frame received:", ret)

if ret:
    print("4. Frame size:", frame.shape)

    cv2.namedWindow("Laptop Camera", cv2.WINDOW_NORMAL)
    cv2.imshow("Laptop Camera", frame)

    print("5. Camera window should be visible now.")
    print("Press any key inside the camera window to close.")

    cv2.waitKey(0)

else:
    print("Could not read frame")

cap.release()
cv2.destroyAllWindows()

print("6. Program finished.")