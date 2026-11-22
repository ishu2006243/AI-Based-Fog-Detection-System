# Hardware Setup – Fog Vision Assistance System

## 1. Hardware Components
- ESP32 Development Board
- Camera Module
- Air Blower
- Connecting Wires
- Power Supply

## 2. Working Principle

The hardware system uses an ESP32 to control an air blower for camera lens cleaning.

1. The camera captures images from the environment.
2. The fog detection system processes the captured images.
3. When camera cleaning is required, the Python application communicates with the ESP32 through serial communication.
4. The ESP32 controls the air blower.
5. Air is directed towards the camera lens to remove dust or moisture.
6. The camera continues capturing images after cleaning.

## 3. Software and Hardware Communication

- Python handles serial communication.
- `serial_controller.py` manages communication between the application and ESP32.
- ESP32 controls the air-blower mechanism.

## 4. Firmware Availability

The ESP32 firmware was programmed separately. Its source code is currently unavailable in this repository.

## 5. Expected Outcome

The air-blower mechanism helps maintain camera lens cleanliness and supports reliable image capture in foggy environments.