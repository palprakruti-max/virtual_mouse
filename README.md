# 🖱️ Gesture-Controlled Virtual Mouse

An AI-powered Virtual Mouse built with Python that allows you to control your computer cursor using hand gestures. It utilizes Computer Vision to track your index finger and thumb, providing a touchless interaction experience.

## 🚀 Features
* **Real-time Tracking:** Uses MediaPipe to track hand landmarks with high precision.
* **Gesture-to-Click:** Perform a pinch gesture (index finger + thumb) to simulate a mouse click.
* **Optimized Performance:** Built with Region of Interest (ROI) tracking to ensure stability and smooth movement at a distance (30-40cm).
* **Low Latency:** Optimized for modern CPUs (e.g., Intel i5 12th Gen) with frame-skipping logic to prevent lag.
* **Touchless Interaction:** Perfect for presentations, accessibility, or just a cool hands-free experience.

## 🛠 Tech Stack
* **Python**
* **OpenCV** (Computer Vision)
* **MediaPipe** (Hand Tracking)
* **PyAutoGUI** (Mouse Control)
* **NumPy** (Mathematical processing)

## 📋 Prerequisites
Ensure you have Python installed on your system. It is recommended to use a stable version (Python 3.9 - 3.12).

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/virtual-mouse.git](https://github.com/yourusername/virtual-mouse.git)
   cd virtual-mouse
