import cv2
import torch
import numpy as np

from model import AODnet

from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights
)


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
        new_state_dict[key.replace("module.", "")] = value

    aod_model.load_state_dict(new_state_dict)

aod_model.eval()

print("AOD-Net loaded successfully!")


# ==========================================
# LOAD SSD MOBILE NET
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
# SYNTHETIC FOG
# ==========================================

def add_fog(frame, strength):

    image = frame.astype(
        np.float32
    ) / 255.0

    white = np.ones_like(image)

    blurred = cv2.GaussianBlur(
        image,
        (21, 21),
        0
    )

    foggy = (
        image * (1 - strength)
        + blurred * strength * 0.5
        + white * strength * 0.5
    )

    foggy = np.clip(
        foggy,
        0,
        1
    )

    return (
        foggy * 255
    ).astype(np.uint8)


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
print("AI Fog Vision System Started")
print()
print("1 = Light Fog")
print("2 = Medium Fog")
print("3 = Dense Fog")
print("0 = Clear")
print("Q = Quit")


fog_strength = 0.0


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # --------------------------------------
    # RESIZE
    # --------------------------------------

    frame = cv2.resize(
        frame,
        (320, 240)
    )


    # --------------------------------------
    # ADD SYNTHETIC FOG
    # --------------------------------------

    foggy = add_fog(
        frame,
        fog_strength
    )


    # --------------------------------------
    # AOD-NET
    # --------------------------------------

    rgb = cv2.cvtColor(
        foggy,
        cv2.COLOR_BGR2RGB
    )

    image = (
        rgb.astype(np.float32)
        / 255.0
    )

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).to(device)


    with torch.no_grad():

        enhanced_tensor = aod_model(
            tensor
        )


    enhanced_tensor = torch.clamp(
        enhanced_tensor,
        0,
        1
    )

    enhanced = (
        enhanced_tensor
        .squeeze(0)
        .cpu()
        .numpy()
    )

    enhanced = np.transpose(
        enhanced,
        (1, 2, 0)
    )

    enhanced = (
        enhanced * 255
    ).astype(np.uint8)

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------
    # SSD MOBILE NET
    # --------------------------------------

    rgb_enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_BGR2RGB
    )

    ssd_image = torch.from_numpy(
        rgb_enhanced
    ).permute(2, 0, 1).float() / 255.0

    ssd_image = ssd_image.to(device)


    with torch.no_grad():

        prediction = ssd_model(
            [ssd_image]
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

        if confidence < 0.30:
            continue


        x1, y1, x2, y2 = (
            box.int()
            .cpu()
            .tolist()
        )


        class_name = categories[
            label.item()
        ]


        cv2.rectangle(
            enhanced,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        text = (
            f"{class_name} "
            f"{confidence:.2f}"
        )


        cv2.putText(
            enhanced,
            text,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )


    # ======================================
    # DISPLAY
    # ======================================

    original = frame.copy()

    cv2.putText(
        original,
        "ORIGINAL",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )


    cv2.putText(
        foggy,
        f"FOG {fog_strength:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        enhanced,
        "AOD-NET + SSD",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )


    # Three views

    top = np.hstack(
        (
            original,
            foggy
        )
    )

    bottom = np.hstack(
        (
            enhanced,
            enhanced.copy()
        )
    )


    combined = np.vstack(
        (
            top,
            bottom
        )
    )


    cv2.imshow(
        "AI Fog Vision System",
        combined
    )


    # ======================================
    # KEYBOARD
    # ======================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("0"):
        fog_strength = 0.0

    elif key == ord("1"):
        fog_strength = 0.3

    elif key == ord("2"):
        fog_strength = 0.6

    elif key == ord("3"):
        fog_strength = 0.85

    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("AI Fog Vision System stopped.")