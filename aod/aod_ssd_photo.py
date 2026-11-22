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
# LOAD FOGGY PHOTO
# ==========================================

image_path = "../fog_test.jpg"

frame = cv2.imread(image_path)

if frame is None:
    print("ERROR: Could not find fog_test.jpg")
    print("Expected location:")
    print(image_path)
    exit()

print("Foggy image loaded successfully!")


# ==========================================
# AOD-NET DEHAZING
# ==========================================

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
).unsqueeze(0).to(device)


with torch.no_grad():

    enhanced_tensor = aod_model(tensor)


enhanced_tensor = torch.clamp(
    enhanced_tensor,
    0,
    1
)


# Tensor → image

enhanced = enhanced_tensor.squeeze(0)
enhanced = enhanced.cpu().numpy()

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


print("AOD-Net enhancement completed!")


# ==========================================
# SSD MOBILE NET DETECTION
# ==========================================

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


print("SSD MobileNet detection completed!")


# ==========================================
# DRAW DETECTIONS
# ==========================================

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
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )


# ==========================================
# DISPLAY
# ==========================================

original = cv2.resize(
    frame,
    (320, 240)
)

cv2.putText(
    original,
    "ORIGINAL FOG",
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 0, 255),
    2
)

cv2.putText(
    enhanced,
    "AOD-NET + SSD",
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 255, 0),
    2
)

combined = np.hstack(
    (original, enhanced)
)

cv2.imwrite(
    "../fog_result.jpg",
    combined
)

cv2.imshow(
    "Fog Vision Test",
    combined
)

print()
print("Result saved as:")
print("../fog_result.jpg")
print()
print("Press any key to close.")

cv2.waitKey(0)

cv2.destroyAllWindows()