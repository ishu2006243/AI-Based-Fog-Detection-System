import cv2
import numpy as np


def generate_fog(frame, fog_strength):

    image = frame.astype(np.float32) / 255.0

    h, w = image.shape[:2]

    # -----------------------------------
    # Approximate scene depth
    # -----------------------------------

    # Farther regions receive more fog
    depth = np.linspace(
        0.0,
        1.0,
        h
    ).reshape(h, 1)

    depth = np.repeat(
        depth,
        w,
        axis=1
    )

    # -----------------------------------
    # Atmospheric transmission
    # -----------------------------------

    beta = 2.5 * fog_strength

    transmission = np.exp(
        -beta * depth
    )

    transmission = np.expand_dims(
        transmission,
        axis=2
    )

    # -----------------------------------
    # Atmospheric light
    # -----------------------------------

    atmospheric_light = np.ones_like(
        image
    ) * 0.95

    # -----------------------------------
    # Atmospheric scattering model
    #
    # Foggy = Clear*T + A*(1-T)
    # -----------------------------------

    foggy = (
        image * transmission
        +
        atmospheric_light
        * (1 - transmission)
    )

    foggy = np.clip(
        foggy,
        0,
        1
    )

    return (
        foggy * 255
    ).astype(np.uint8)


# =====================================
# CAMERA
# =====================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("Camera could not be opened")
    exit()


print("Realistic Fog Simulation")
print()
print("0 = Clear")
print("1 = Light Fog")
print("2 = Medium Fog")
print("3 = Dense Fog")
print("Q = Quit")


fog_strength = 0.0


# =====================================
# LOOP
# =====================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    frame = cv2.resize(
        frame,
        (640, 480)
    )


    foggy = generate_fog(
        frame,
        fog_strength
    )


    # ---------------------------------
    # Display
    # ---------------------------------

    original = frame.copy()

    cv2.putText(
        original,
        "CLEAR IMAGE",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    cv2.putText(
        foggy,
        f"FOG LEVEL: {fog_strength:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )


    combined = np.hstack(
        (
            original,
            foggy
        )
    )


    cv2.imshow(
        "Atmospheric Fog Simulation",
        combined
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("0"):

        fog_strength = 0.0


    elif key == ord("1"):

        fog_strength = 0.25


    elif key == ord("2"):

        fog_strength = 0.50


    elif key == ord("3"):

        fog_strength = 0.75


    elif key == ord("q"):

        break


cap.release()

cv2.destroyAllWindows()