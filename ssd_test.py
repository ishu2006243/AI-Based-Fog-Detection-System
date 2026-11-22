import cv2
import torch
import torchvision
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights
)


# -----------------------------
# Load SSD MobileNet
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT

print("Loading SSD MobileNet...")

model = ssdlite320_mobilenet_v3_large(
    weights=weights
)

model = model.to(device)
model.eval()

print("SSD MobileNet loaded successfully!")


# COCO class names
categories = weights.meta["categories"]


# -----------------------------
# Camera
# -----------------------------

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    print("Camera could not be opened")
    exit()

print("SSD MobileNet camera started.")
print("Press Q to quit.")


# -----------------------------
# Detection
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame")
        break

    # Resize for display
    display = cv2.resize(frame, (640, 480))

    # BGR → RGB
    rgb = cv2.cvtColor(
        display,
        cv2.COLOR_BGR2RGB
    )

    # Convert image to tensor
    image = torch.from_numpy(
        rgb
    ).permute(2, 0, 1).float() / 255.0

    image = image.to(device)

    # Add batch dimension
    image = image.unsqueeze(0)


    # Run SSD MobileNet
    with torch.no_grad():

        prediction = model(image)[0]


    # -----------------------------
    # Draw detections
    # -----------------------------

    boxes = prediction["boxes"]
    labels = prediction["labels"]
    scores = prediction["scores"]


    for box, label, score in zip(
        boxes,
        labels,
        scores
    ):

        if score < 0.50:
            continue

        x1, y1, x2, y2 = box.int().cpu().tolist()

        class_name = categories[
            label.item()
        ]

        confidence = score.item()


        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        text = f"{class_name} {confidence:.2f}"


        cv2.putText(
            display,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    cv2.imshow(
        "SSD MobileNet - Object Detection",
        display
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("SSD MobileNet stopped.")