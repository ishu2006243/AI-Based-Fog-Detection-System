import cv2
import torch
import numpy as np
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights
)

from model import AODnet


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# LOAD AOD-NET
# ==========================================

print("Loading AOD-Net...")

weight_path = "model_pretrained/AOD_net_epoch_relu_10.pth"

checkpoint = torch.load(
    weight_path,
    map_location=device,
    weights_only=False
)

if isinstance(checkpoint, torch.nn.Module):

    aod_model = checkpoint.to(device)

else:

    aod_model = AODnet().to(device)

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    new_state_dict = {}

    for key, value in checkpoint.items():
        key = key.replace("module.", "")
        new_state_dict[key] = value

    aod_model.load_state_dict(new_state_dict)


aod_model.eval()

print("AOD-Net loaded successfully!")


# ==========================================
# LOAD SSD MOBILENET
# ==========================================

print("Loading SSD MobileNet...")

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT

ssd_model = ssdlite320_mobilenet_v3_large(
    weights=weights
)

ssd_model = ssd_model.to(device)
ssd_model.eval()

categories = weights.meta["categories"]

print("SSD MobileNet loaded successfully!")


# ==========================================
# AOD-NET DEHAZING
# ==========================================

def dehaze(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    rgb = cv2.resize(
        rgb,
        (320, 240)
    )

    image = rgb.astype(
        np.float32
    ) / 255.0

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():

        output = aod_model(tensor)

    output = torch.clamp(
        output,
        0,
        1
    )

    output = output.squeeze(0)
    output = output.cpu().numpy()

    output = np.transpose(
        output,
        (1, 2, 0)
    )

    output = (
        output * 255
    ).astype(np.uint8)

    output = cv2.cvtColor(
        output,
        cv2.COLOR_RGB2BGR
    )

    return output


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("Camera could not be opened")
    exit()


print()
print("AI Fog Vision started!")
print("AOD-Net + SSD MobileNet")
print("Press Q to quit.")


# ==========================================
# LIVE PIPELINE
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not read camera frame")
        break


    # --------------------------------------
    # AOD-NET
    # --------------------------------------

    enhanced = dehaze(frame)


    # --------------------------------------
    # SSD MOBILE NET
    # --------------------------------------

    rgb = cv2.cvtColor(
        enhanced,
        cv2.COLOR_BGR2RGB
    )

    image = torch.from_numpy(
        rgb
    ).permute(2, 0, 1).float() / 255.0

    image = image.to(device)


    with torch.no_grad():

        prediction = ssd_model(
            [image]
        )[0]


    # --------------------------------------
    # DRAW DETECTIONS
    # --------------------------------------

    for box, label, score in zip(
        prediction["boxes"],
        prediction["labels"],
        prediction["scores"]
    ):

        confidence = score.item()

        if confidence < 0.50:
            continue


        x1, y1, x2, y2 = (
            box.int()
            .cpu()
            .tolist()
        )


        class_name = categories[
            label.item()
        ]


        # Draw bounding box

        cv2.rectangle(
            enhanced,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # Detection text

        text = (
            f"{class_name} "
            f"{confidence:.2f}"
        )


        cv2.putText(
            enhanced,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )


    # --------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------

    original = cv2.resize(
        frame,
        (320, 240)
    )


    cv2.putText(
        original,
        "ORIGINAL",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )


    cv2.putText(
        enhanced,
        "AOD-NET + SSD",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # --------------------------------------
    # SIDE-BY-SIDE DISPLAY
    # --------------------------------------

    combined = np.hstack(
        (
            original,
            enhanced
        )
    )


    cv2.imshow(
        "AI Fog Vision System",
        combined
    )


    # --------------------------------------
    # QUIT
    # --------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("AI Fog Vision stopped.")