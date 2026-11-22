import cv2
import torch
import numpy as np

from model import AODnet


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# Load AOD-Net
# -----------------------------

model = AODnet().to(device)

weight_path = "model_pretrained/AOD_net_epoch_relu_10.pth"

print("Loading AOD-Net...")

checkpoint = torch.load(
    weight_path,
    map_location=device,
    weights_only=False
)

# Your checkpoint is a complete AOD-Net model
if isinstance(checkpoint, torch.nn.Module):

    model = checkpoint.to(device)

else:

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    new_state_dict = {}

    for key, value in checkpoint.items():

        key = key.replace("module.", "")
        new_state_dict[key] = value

    model.load_state_dict(new_state_dict)


model.eval()

print("AOD-Net loaded successfully!")


# -----------------------------
# Dehazing function
# -----------------------------

def dehaze(frame):

    # BGR → RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Reduce resolution for CPU speed
    rgb = cv2.resize(
        rgb,
        (320, 240)
    )

    # Convert to float
    image = rgb.astype(
        np.float32
    ) / 255.0

    # HWC → CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # NumPy → Tensor
    tensor = torch.from_numpy(
        image
    ).unsqueeze(0)

    tensor = tensor.to(device)

    # AOD-Net inference
    with torch.no_grad():

        output = model(tensor)

    # Keep range 0–1
    output = torch.clamp(
        output,
        0,
        1
    )

    # Tensor → NumPy
    output = output.squeeze(0)
    output = output.cpu().numpy()

    # CHW → HWC
    output = np.transpose(
        output,
        (1, 2, 0)
    )

    # 0–1 → 0–255
    output = (
        output * 255
    ).astype(np.uint8)

    # RGB → BGR
    output = cv2.cvtColor(
        output,
        cv2.COLOR_RGB2BGR
    )

    return output


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


print("AOD-Net camera started.")
print("Press Q to quit.")


# -----------------------------
# Live processing
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not read camera frame")
        break

    # AOD-Net enhancement
    enhanced = dehaze(frame)

    # Original
    original = cv2.resize(
        frame,
        (320, 240)
    )

    # Labels
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
        "AOD-NET ENHANCED",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Side-by-side
    combined = np.hstack(
        (original, enhanced)
    )

    cv2.imshow(
        "AI Fog Vision - AOD-Net",
        combined
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()