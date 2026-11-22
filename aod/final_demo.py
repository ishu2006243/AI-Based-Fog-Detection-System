# ==============================================================
# AI FOG VISION SAFETY SYSTEM
# ENHANCED AOD-NET + SSD MOBILE NET
# ==============================================================

import cv2
import torch
import numpy as np

from model import AODnet

from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights
)


# ==============================================================
# DEVICE
# ==============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==============================================================
# LOAD AOD-NET
# ==============================================================

print("Loading AOD-Net...")

weight_path = "model_pretrained/AOD_net_epoch_relu_10.pth"

checkpoint = torch.load(
    weight_path,
    map_location=device,
    weights_only=False
)


if isinstance(checkpoint, torch.nn.Module):

    # Complete pretrained model
    aod_model = checkpoint.to(device)

else:

    # State dictionary
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


# ==============================================================
# LOAD SSD MOBILE NET
# ==============================================================

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


# ==============================================================
# REALISTIC ATMOSPHERIC FOG GENERATOR
# ==============================================================

def generate_fog(frame, strength):

    # ----------------------------------------------------------
    # Convert image to floating point
    # ----------------------------------------------------------

    image = (
        frame.astype(np.float32)
        / 255.0
    )

    h, w = image.shape[:2]


    # ----------------------------------------------------------
    # Depth map
    # ----------------------------------------------------------

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


    # ----------------------------------------------------------
    # Fog density
    # ----------------------------------------------------------

    beta = 3.5 * strength


    # ----------------------------------------------------------
    # Transmission
    # ----------------------------------------------------------

    transmission = np.exp(
        -beta * depth
    )

    transmission = np.expand_dims(
        transmission,
        axis=2
    )


    # ----------------------------------------------------------
    # Atmospheric light
    # ----------------------------------------------------------

    atmospheric_light = (
        np.ones_like(image)
        * 0.95
    )


    # ----------------------------------------------------------
    # Atmospheric scattering model
    #
    # I = J*T + A*(1-T)
    # ----------------------------------------------------------

    foggy = (
        image * transmission
        +
        atmospheric_light
        * (1 - transmission)
    )


    # ----------------------------------------------------------
    # Clip
    # ----------------------------------------------------------

    foggy = np.clip(
        foggy,
        0.0,
        1.0
    )


    return (
        foggy * 255
    ).astype(np.uint8)


# ==============================================================
# STRONG ENHANCED AOD-NET DEHAZING
# ==============================================================

def dehaze(frame):

    # ==========================================================
    # 1. BGR -> RGB
    # ==========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ==========================================================
    # 2. Normalize image
    # ==========================================================

    image = (
        rgb.astype(np.float32)
        / 255.0
    )

    image = np.clip(
        image,
        0.0,
        1.0
    )


    # ==========================================================
    # 3. Percentile-based input normalization
    # ==========================================================

    low = np.percentile(
        image,
        1
    )

    high = np.percentile(
        image,
        99
    )


    image = (
        image - low
    ) / (
        high - low + 1e-6
    )


    image = np.clip(
        image,
        0.0,
        1.0
    )


    # ==========================================================
    # 4. Convert to PyTorch tensor
    # ==========================================================

    image = np.transpose(
        image,
        (2, 0, 1)
    )


    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).to(device)


    # ==========================================================
    # 5. AOD-NET INFERENCE
    # ==========================================================

    with torch.no_grad():

        output = aod_model(
            tensor
        )


    # ==========================================================
    # 6. Clamp output
    # ==========================================================

    output = torch.clamp(
        output,
        0.0,
        1.0
    )


    # ==========================================================
    # 7. Tensor -> NumPy
    # ==========================================================

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
        output * 255.0
    ).astype(np.uint8)


    # ==========================================================
    # 8. RGB -> BGR
    # ==========================================================

    output = cv2.cvtColor(
        output,
        cv2.COLOR_RGB2BGR
    )


    # ==========================================================
    # 9. ATMOSPHERIC VEIL SUPPRESSION
    # ==========================================================

    lab = cv2.cvtColor(
        output,
        cv2.COLOR_BGR2LAB
    )


    l, a, b = cv2.split(
        lab
    )


    # ----------------------------------------------------------
    # Estimate large-scale illumination
    # ----------------------------------------------------------

    illumination = cv2.GaussianBlur(
        l,
        (0, 0),
        25
    )


    # ----------------------------------------------------------
    # Strong atmospheric veil suppression
    # ----------------------------------------------------------

    refined_l = cv2.addWeighted(
        l,
        1.45,
        illumination,
        -0.45,
        0
    )


    refined_l = np.clip(
        refined_l,
        0,
        255
    ).astype(np.uint8)


    # ==========================================================
    # 10. STRONG LOCAL CONTRAST
    # ==========================================================

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )


    refined_l = clahe.apply(
        refined_l
    )


    # ==========================================================
    # 11. CONTRAST STRETCHING
    # ==========================================================

    p2, p98 = np.percentile(
        refined_l,
        (2, 98)
    )


    if p98 > p2:

        refined_l = (
            refined_l.astype(np.float32)
            - p2
        ) / (
            p98 - p2
        )


        refined_l = np.clip(
            refined_l,
            0.0,
            1.0
        )


        refined_l = (
            refined_l * 255
        ).astype(np.uint8)


    # ==========================================================
    # 12. REBUILD LAB IMAGE
    # ==========================================================

    refined = cv2.merge(
        (
            refined_l,
            a,
            b
        )
    )


    refined = cv2.cvtColor(
        refined,
        cv2.COLOR_LAB2BGR
    )


    # ==========================================================
    # 13. COLOR RECOVERY
    # ==========================================================

    hsv = cv2.cvtColor(
        refined,
        cv2.COLOR_BGR2HSV
    )


    h, s, v = cv2.split(
        hsv
    )


    # ----------------------------------------------------------
    # Slight saturation recovery
    # ----------------------------------------------------------

    s = np.clip(
        s.astype(np.float32) * 1.15,
        0,
        255
    ).astype(np.uint8)


    hsv = cv2.merge(
        (
            h,
            s,
            v
        )
    )


    refined = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


    # ==========================================================
    # 14. GAMMA CORRECTION
    # ==========================================================

    gamma = 0.75


    lookup = np.array([
        (
            (i / 255.0) ** gamma
        ) * 255.0
        for i in range(256)
    ]).astype(np.uint8)


    refined = cv2.LUT(
        refined,
        lookup
    )


    # ==========================================================
    # 15. DETAIL ENHANCEMENT
    # ==========================================================

    smooth = cv2.GaussianBlur(
        refined,
        (0, 0),
        1.2
    )


    refined = cv2.addWeighted(
        refined,
        1.35,
        smooth,
        -0.35,
        0
    )


    # ==========================================================
    # 16. FINAL CLAHE
    # ==========================================================

    lab = cv2.cvtColor(
        refined,
        cv2.COLOR_BGR2LAB
    )


    l, a, b = cv2.split(
        lab
    )


    clahe_final = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )


    l = clahe_final.apply(
        l
    )


    refined = cv2.merge(
        (
            l,
            a,
            b
        )
    )


    refined = cv2.cvtColor(
        refined,
        cv2.COLOR_LAB2BGR
    )


    # ==========================================================
    # 17. FINAL CLIPPING
    # ==========================================================

    refined = np.clip(
        refined,
        0,
        255
    ).astype(np.uint8)


    return refined


# ==============================================================
# FINAL VISIBILITY ENHANCEMENT
# ==============================================================

def improve_visibility(image):

    # ----------------------------------------------------------
    # LAB conversion
    # ----------------------------------------------------------

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )


    l, a, b = cv2.split(
        lab
    )


    # ----------------------------------------------------------
    # Mild final CLAHE
    # ----------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=1.2,
        tileGridSize=(8, 8)
    )


    l = clahe.apply(
        l
    )


    # ----------------------------------------------------------
    # Rebuild image
    # ----------------------------------------------------------

    enhanced = cv2.merge(
        (
            l,
            a,
            b
        )
    )


    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )


    # ==========================================================
    # Mild gamma correction
    # ==========================================================

    gamma = 0.90


    lookup = np.array([
        (
            (i / 255.0) ** gamma
        ) * 255.0
        for i in range(256)
    ]).astype(np.uint8)


    enhanced = cv2.LUT(
        enhanced,
        lookup
    )


    # ----------------------------------------------------------
    # Final clipping
    # ----------------------------------------------------------

    enhanced = np.clip(
        enhanced,
        0,
        255
    ).astype(np.uint8)


    return enhanced


# ==============================================================
# SSD MOBILE NET OBJECT DETECTION
# ==============================================================

def detect_objects(image):

    # ----------------------------------------------------------
    # BGR -> RGB
    # ----------------------------------------------------------

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # ----------------------------------------------------------
    # Convert to tensor
    # ----------------------------------------------------------

    tensor = torch.from_numpy(
        rgb
    ).permute(
        2,
        0,
        1
    ).float() / 255.0


    tensor = tensor.to(device)


    # ==========================================================
    # SSD INFERENCE
    # ==========================================================

    with torch.no_grad():

        prediction = ssd_model(
            [tensor]
        )[0]


    # ----------------------------------------------------------
    # Copy image
    # ----------------------------------------------------------

    result = image.copy()


    detected_count = 0


    # ==========================================================
    # Detection results
    # ==========================================================

    for box, label, score in zip(
        prediction["boxes"],
        prediction["labels"],
        prediction["scores"]
    ):

        confidence = score.item()


        # ------------------------------------------------------
        # Confidence threshold
        # ------------------------------------------------------

        if confidence < 0.30:
            continue


        # ------------------------------------------------------
        # Bounding box
        # ------------------------------------------------------

        x1, y1, x2, y2 = (
            box.int()
            .cpu()
            .tolist()
        )


        # ------------------------------------------------------
        # Class name
        # ------------------------------------------------------

        class_name = categories[
            label.item()
        ]


        # ======================================================
        # Draw bounding box
        # ======================================================

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # ======================================================
        # Detection label
        # ======================================================

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


# ==============================================================
# CAMERA
# ==============================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


if not cap.isOpened():

    print(
        "Camera could not be opened"
    )

    exit()


# ==============================================================
# INFORMATION
# ==============================================================

print()
print("================================================")
print("        AI FOG VISION SAFETY SYSTEM")
print("================================================")
print()
print("Processing Pipeline:")
print()
print("Camera")
print("   ↓")
print("Atmospheric Fog Simulation")
print("   ↓")
print("Input Normalization")
print("   ↓")
print("AOD-Net Dehazing")
print("   ↓")
print("Atmospheric Veil Suppression")
print("   ↓")
print("Contrast Enhancement")
print("   ↓")
print("Color Recovery")
print("   ↓")
print("Gamma Correction")
print("   ↓")
print("Detail Enhancement")
print("   ↓")
print("SSD MobileNet")
print("   ↓")
print("Object Detection")
print()
print("------------------------------------------------")
print("Controls")
print("------------------------------------------------")
print("0 = Clear")
print("1 = Light Fog     (30%)")
print("2 = Medium Fog    (60%)")
print("3 = Dense Fog     (90%)")
print("Q = Quit")
print("================================================")
print()


# ==============================================================
# INITIAL FOG LEVEL
# ==============================================================

fog_strength = 0.0


# ==============================================================
# MAIN LOOP
# ==============================================================

while True:

    # ==========================================================
    # CAPTURE FRAME
    # ==========================================================

    ret, frame = cap.read()


    if not ret:

        print(
            "Camera frame unavailable"
        )

        break


    # ==========================================================
    # RESIZE
    # ==========================================================

    frame = cv2.resize(
        frame,
        (320, 240)
    )


    # ==========================================================
    # STAGE 1 — FOG GENERATION
    # ==========================================================

    foggy = generate_fog(
        frame,
        fog_strength
    )


    # ==========================================================
    # STAGE 2 — AOD-NET DEHAZING
    # ==========================================================

    aod_output = dehaze(
        foggy
    )


    # ==========================================================
    # STAGE 3 — VISIBILITY REFINEMENT
    # ==========================================================

    enhanced = improve_visibility(
        aod_output
    )


    # ==========================================================
    # STAGE 4 — SSD MOBILE NET
    # ==========================================================

    detected, count = detect_objects(
        enhanced
    )


    # ==========================================================
    # PANEL 1 — CLEAR IMAGE
    # ==========================================================

    clear_panel = frame.copy()


    cv2.putText(
        clear_panel,
        "CLEAR IMAGE",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )


    # ==========================================================
    # PANEL 2 — FOGGY IMAGE
    # ==========================================================

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


    # ==========================================================
    # PANEL 3 — AOD-NET + SSD
    # ==========================================================

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


    # ==========================================================
    # THREE PANEL DISPLAY
    # ==========================================================

    combined = np.hstack(
        (
            clear_panel,
            fog_panel,
            detected
        )
    )


    cv2.imshow(
        "AI Fog Vision System",
        combined
    )


    # ==========================================================
    # KEYBOARD CONTROL
    # ==========================================================

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


# ==============================================================
# EXIT
# ==============================================================

cap.release()

cv2.destroyAllWindows()


print()
print(
    "AI Fog Vision System stopped."
)