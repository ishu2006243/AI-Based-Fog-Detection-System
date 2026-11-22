import cv2
import torch
import numpy as np

from model import AODnet

from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights
)

# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==================================================
# LOAD AOD-NET
# ==================================================

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

        new_state_dict[
            key.replace("module.", "")
        ] = value

    aod_model.load_state_dict(
        new_state_dict
    )

aod_model.eval()

print("AOD-Net loaded successfully!")


# ==================================================
# LOAD SSD MOBILE NET
# ==================================================

print("Loading SSD MobileNet...")

weights = (
    SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
)

ssd_model = (
    ssdlite320_mobilenet_v3_large(
        weights=weights
    )
)

ssd_model = ssd_model.to(device)

ssd_model.eval()

categories = weights.meta["categories"]

print("SSD MobileNet loaded successfully!")


# ==================================================
# REALISTIC ATMOSPHERIC FOG GENERATOR
# ==================================================

def generate_fog(frame, strength):

    # Convert image to floating point
    image = (
        frame.astype(np.float32)
        / 255.0
    )

    h, w = image.shape[:2]

    # ----------------------------------------------
    # Depth map
    # ----------------------------------------------

    depth = np.linspace(
        1.0,
        0.0,
        h
    ).reshape(h, 1)

    depth = np.repeat(
        depth,
        w,
        axis=1
    )

    # ----------------------------------------------
    # Atmospheric scattering
    # ----------------------------------------------

    beta = 3.5 * strength

    transmission = np.exp(
        -beta * depth
    )

    transmission = np.expand_dims(
        transmission,
        axis=2
    )

    # ----------------------------------------------
    # Atmospheric light
    # ----------------------------------------------

    atmospheric_light = (
        np.ones_like(image)
        * 0.95
    )

    # ----------------------------------------------
    # Fog formation
    #
    # I = J*T + A*(1-T)
    # ----------------------------------------------

    foggy = (
        image * transmission
        +
        atmospheric_light
        * (1 - transmission)
    )

    foggy = np.clip(
        foggy,
        0.0,
        1.0
    )

    return (
        foggy * 255
    ).astype(np.uint8)


# ==================================================
# AOD-NET DEHAZING
# ==================================================

def dehaze(frame):

    # ----------------------------------------------
    # BGR → RGB
    # ----------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # ----------------------------------------------
    # Normalize
    # ----------------------------------------------

    image = (
        rgb.astype(np.float32)
        / 255.0
    )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    # ----------------------------------------------
    # Convert image to tensor
    # ----------------------------------------------

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0)

    tensor = tensor.to(device)

    # ----------------------------------------------
    # AOD-Net inference
    # ----------------------------------------------

    with torch.no_grad():

        output = aod_model(
            tensor
        )

    # ----------------------------------------------
    # Clamp output
    # ----------------------------------------------

    output = torch.clamp(
        output,
        0.0,
        1.0
    )

    # ----------------------------------------------
    # Tensor → NumPy
    # ----------------------------------------------

    output = (
        output
        .squeeze(0)
        .cpu()
        .numpy()
    )

    output = np.transpose(
        output,
        (1, 2, 0)
    )

    output = (
        output * 255
    ).astype(np.uint8)

    # ----------------------------------------------
    # RGB → BGR
    # ----------------------------------------------

    output = cv2.cvtColor(
        output,
        cv2.COLOR_RGB2BGR
    )

    # ----------------------------------------------
    # Mild CLAHE enhancement
    # ----------------------------------------------

    lab = cv2.cvtColor(
        output,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    output = cv2.merge(
        (l, a, b)
    )

    output = cv2.cvtColor(
        output,
        cv2.COLOR_LAB2BGR
    )

    return output


# ==================================================
# VISIBILITY ENHANCEMENT
# ==================================================

def improve_visibility(image):

    # ----------------------------------------------
    # LAB conversion
    # ----------------------------------------------

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(
        lab
    )

    # ----------------------------------------------
    # CLAHE
    # ----------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge(
        (l, a, b)
    )

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )

    # ----------------------------------------------
    # Mild gamma correction
    # ----------------------------------------------

    gamma = 0.85

    lookup = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype(np.uint8)

    enhanced = cv2.LUT(
        enhanced,
        lookup
    )

    return enhanced


# ==================================================
# SSD MOBILE NET OBJECT DETECTION
# ==================================================

def detect_objects(image):

    # ----------------------------------------------
    # BGR → RGB
    # ----------------------------------------------

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # ----------------------------------------------
    # Convert image to tensor
    # ----------------------------------------------

    tensor = torch.from_numpy(
        rgb
    ).permute(
        2,
        0,
        1
    ).float() / 255.0

    tensor = tensor.to(device)

    # ----------------------------------------------
    # SSD inference
    # ----------------------------------------------

    with torch.no_grad():

        prediction = ssd_model(
            [tensor]
        )[0]

    result = image.copy()

    detected_count = 0

    # ----------------------------------------------
    # Detection results
    # ----------------------------------------------

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

        # ------------------------------------------
        # Bounding box
        # ------------------------------------------

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ------------------------------------------
        # Label
        # ------------------------------------------

        text = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.putText(
            result,
            text,
            (
                x1,
                max(y1 - 8, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )

        detected_count += 1

    return result, detected_count


# ==================================================
# CAMERA
# ==================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print(
        "Camera could not be opened"
    )

    exit()


# ==================================================
# INFORMATION
# ==================================================

print()
print("==============================================")
print("       AI FOG VISION SAFETY SYSTEM")
print("==============================================")
print()
print("0 = Clear")
print("1 = Light Fog     (30%)")
print("2 = Medium Fog    (60%)")
print("3 = Dense Fog     (90%)")
print("Q = Quit")
print()


# ==================================================
# INITIAL FOG LEVEL
# ==================================================

fog_strength = 0.0


# ==================================================
# CREATE LARGE DISPLAY WINDOW
# ==================================================

cv2.namedWindow(
    "AI Fog Vision System",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "AI Fog Vision System",
    1280,
    520
)


# ==================================================
# MAIN LOOP
# ==================================================

while True:

    # ----------------------------------------------
    # Capture frame
    # ----------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print(
            "Camera frame unavailable"
        )

        break

    # ----------------------------------------------
    # Resize for AI processing
    # ----------------------------------------------

    frame = cv2.resize(
        frame,
        (320, 240)
    )

    # ==============================================
    # STAGE 1 — FOG GENERATION
    # ==============================================

    foggy = generate_fog(
        frame,
        fog_strength
    )

    # ==============================================
    # STAGE 2 — AOD-NET
    # ==============================================

    aod_output = dehaze(
        foggy
    )

    # ==============================================
    # STAGE 3 — VISIBILITY ENHANCEMENT
    # ==============================================

    enhanced = improve_visibility(
        aod_output
    )

    # ==============================================
    # STAGE 4 — SSD MOBILE NET
    # ==============================================

    detected, count = detect_objects(
        enhanced
    )

    # ==============================================
    # PANEL 1 — FOGGY IMAGE
    # ==============================================

    fog_panel = foggy.copy()

    cv2.putText(
        fog_panel,
        f"FOG {int(fog_strength * 100)}%",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        fog_panel,
        "ATMOSPHERIC FOG",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1
    )

    # ==============================================
    # PANEL 2 — AOD + SSD
    # ==============================================

    cv2.putText(
        detected,
        "AOD-NET + SSD",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 0),
        2
    )

    cv2.putText(
        detected,
        f"Objects: {count}",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 0),
        1
    )

    # ==============================================
    # RESIZE FOR FULL DISPLAY
    # ==============================================

    fog_display = cv2.resize(
        fog_panel,
        (640, 480),
        interpolation=cv2.INTER_NEAREST
    )

    detected_display = cv2.resize(
        detected,
        (640, 480),
        interpolation=cv2.INTER_NEAREST
    )

    # ==============================================
    # TWO PANEL DISPLAY
    # ==============================================

    combined = np.hstack(
        (
            fog_display,
            detected_display
        )
    )

    # ==============================================
    # SHOW FULL-SIZE WINDOW
    # ==============================================

    cv2.imshow(
        "AI Fog Vision System",
        combined
    )

    # ==============================================
    # KEYBOARD CONTROL
    # ==============================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("0"):

        fog_strength = 0.0

    elif key == ord("1"):

        fog_strength = 0.30

    elif key == ord("2"):

        fog_strength = 0.60

    elif key == ord("3"):

        fog_strength = 0.90

    elif key == ord("q"):

        break


# ==================================================
# EXIT
# ==================================================

cap.release()

cv2.destroyAllWindows()

print()
print(
    "AI Fog Vision System stopped."
)