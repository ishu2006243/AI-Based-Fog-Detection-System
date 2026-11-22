# AI-Based Fog Detection and Vision Assistance System

## Project Overview
An AI-based vision assistance system designed to improve visibility in foggy environments using image dehazing, object detection, and an intelligent camera-cleaning mechanism.

## Main Modules

### 1. Atmospheric Optical Dehazing (AOD)
- Enhances images affected by fog.
- Uses an AOD-based deep learning model.
- Improves image visibility and clarity.

### 2. Single Shot Detector (SSD)
- Detects objects in images and video frames.
- Works with the fog-removal pipeline.
- Supports improved object visibility in foggy conditions.

### 3. Fog Detection and Django Web Application
- Python-based fog processing.
- Django backend and web interface.
- Camera and image processing integration.

### 4. ESP32-Based Camera Cleaning Mechanism
- ESP32-based hardware control.
- Air blower used for camera lens cleaning.
- Python serial communication through `serial_controller.py`.

## Technology Stack
- Python
- Django
- HTML, CSS, JavaScript
- AOD-Net
- SSD Object Detection
- ESP32
- Serial Communication

## Repository Structure
- `aod/` – AOD and integrated AOD-SSD implementation.
- `Django_Web_Application/` – Django application and frontend.
- `fog_detection.py` – Fog detection implementation.
- `enhance_camera.py` – Camera enhancement functionality.
- `camera_test.py` – Camera testing.
- `ssd_test.py` – SSD testing.
- `model_pretrained/` – AOD pretrained model inside `aod/`.

## Hardware
- ESP32 development board
- Camera
- Air blower
- Connecting wires and power supply

## Installation

1. Clone the repository:

   `git clone https://github.com/ishu2006243/AI-Based-Fog-Detection-System.git`

2. Open the project directory.

3. Create and activate a Python virtual environment.

4. Install dependencies:

   `pip install -r Django_Web_Application/requirements.txt`

5. Run Django:

   `python Django_Web_Application/manage.py runserver`

## Hardware Firmware
The ESP32 firmware source is maintained separately and is not currently included in this repository. The Python serial communication module is available in the Django application.

## Project Goal
To improve camera visibility and object detection in foggy environments through AI-based image enhancement and an ESP32-controlled air-blower cleaning mechanism.