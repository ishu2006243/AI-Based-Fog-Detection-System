import torch
from model import AODnet

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

weight_path = "model_pretrained/AOD_net_epoch_relu_10.pth"

print("Loading pretrained AOD-Net...")

# This checkpoint is an old PyTorch serialized model.
# weights_only=False is required for this trusted checkpoint.
checkpoint = torch.load(
    weight_path,
    map_location=device,
    weights_only=False
)

print("Checkpoint type:", type(checkpoint))

# If the checkpoint is a complete model
if isinstance(checkpoint, torch.nn.Module):

    model = checkpoint.to(device)

# If the checkpoint contains a state_dict
else:

    model = AODnet().to(device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    new_state_dict = {}

    for key, value in checkpoint.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)

model.eval()

print("AOD-Net pretrained model loaded successfully!")
print("Model is ready.")