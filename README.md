# Raspberry Pi Based Driver Monitoring System

## Overview

This project is a Raspberry Pi-based Driver Monitoring System designed to improve road safety by monitoring driver behavior in real time. The system combines driver authentication, drowsiness detection, and alcohol detection into a single integrated platform.

The system utilizes computer vision techniques, machine learning models, and IoT technologies to continuously monitor the driver's condition and provide immediate alerts when unsafe driving conditions are detected.

A Flutter mobile application is also developed to provide a user-friendly interface for monitoring driver status, authentication results, alerts, and live video streaming.

---

## Features

### Driver Authentication

* Face recognition using face embeddings.
* Verifies authorized drivers before monitoring begins.
* Prevents unauthorized vehicle usage.

### Drowsiness Detection

Detects multiple signs of fatigue:

* Eye Closure Detection

  * Eye Aspect Ratio (EAR)
  * TensorFlow Lite eye-state classification model

* Yawning Detection

  * Mouth Aspect Ratio (MAR)

* Head Bending Detection

  * Nose Length Ratio (NLR)

### Alcohol Detection

* MQ3 Alcohol Gas Sensor integration.
* Detects alcohol concentration from driver's breath.
* Generates warning alerts when alcohol is detected.

### Alert System

* Active buzzer alarm.
* LCD display notifications.
* Mobile application notifications.

### Mobile Application

Developed using Flutter.

Functions include:

* Driver authentication control
* Real-time monitoring
* Alert notifications
* Live video streaming
* Message display
* Raspberry Pi shutdown control

---

## System Architecture

### Hardware Components

* Raspberry Pi 4 Model B
* Logitech C270 HD Webcam
* MQ3 Alcohol Gas Sensor
* MCP3008 ADC Converter
* LCD 1602 (I2C)
* Active Buzzer
* Power Bank

### Software Components

* Python
* OpenCV
* MediaPipe
* TensorFlow Lite
* Flask API
* Raspberry Pi OS
* Flutter
* Google Colab

---

## Detection Methods

### Face Authentication

The system uses face embeddings to generate numerical feature vectors from facial images.

Authentication Process:

1. Capture driver's face.
2. Generate facial embeddings.
3. Compare against registered embeddings in database.
4. Authenticate if similarity exceeds threshold.

### Drowsiness Detection

#### Eye Aspect Ratio (EAR)
Measures eye openness using facial landmarks.

Indicators:
* Continuous eye closure
* Microsleep detection

#### Mouth Aspect Ratio (MAR)
Measures mouth opening.

Indicators:
* Excessive yawning
* Fatigue signs

#### Nose Length Ratio (NLR)
Measures head position changes.

Indicators:
* Head bending downward
* Driver distraction

### Eye-State Classification Model
A TensorFlow Lite model is used to classify:
* Open Eyes
* Closed Eyes

The model is take place after the EAR calculation to improves robustness of the system.

### Alcohol Detection
The MQ3 sensor continuously monitors alcohol vapor concentration.

If the reading exceeds a predefined threshold:
* Alarm is triggered
* LCD warning is displayed
* Mobile application is updated

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/driver-monitoring-system.git
cd driver-monitoring-system
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the System

Start the Flask server:

```bash
python main.py
```

Ensure:

* Webcam is connected.
* MQ3 sensor is connected through MCP3008.
* LCD and buzzer are properly wired.
* Raspberry Pi and mobile application are connected to the same network.

---

## Mobile Application

Navigate to Flutter application folder:

```bash
cd flutter_app
flutter pub get
flutter run
```

Before running:
Update the Raspberry Pi IP address inside the Flutter application.

---

## Results
Kindly check the screenshot folder provided.

The system successfully performs:
* Driver authentication
* Eye closure detection
* Yawning detection
* Head bending detection
* Alcohol detection
* Real-time alerting
* Mobile monitoring

The integration of computer vision, machine learning, sensors, and mobile technology demonstrates a practical IoT-based driver safety solution.

---

## Future Improvements

* Infrared camera support for night driving.
* Liveness detection
* Multiple driver profile management.
* Cloud-based monitoring and logging.
* GPS integration.
* Emergency contact notification.
* Advanced deep learning models for fatigue analysis.

---

## Author
- Chee Hao Jie
- Bachelor of Information Technology (Honours) Computer Engineering
- Universiti Tunku Abdul Rahman (UTAR)
- 2025

---

## License

This project is developed for educational and research purposes.
