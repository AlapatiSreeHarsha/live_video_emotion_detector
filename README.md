# live_video_emotion_detector

# Live People Detection and Emotion Tracking

## Overview
This project leverages real-time computer vision techniques to detect and track people and their emotions using a webcam feed. The system utilizes multiple open-source libraries such as **Streamlit**, **OpenCV**, **Mediapipe**, and **FER (Facial Emotion Recognition)** for real-time detection and analysis of human emotions and counting the number of people in the frame.

## Technologies Used
- **Streamlit**: Used for building the web application interface to display live video feed and interact with the system.
- **OpenCV**: Provides functionality for capturing video frames from the webcam and performing image processing.
- **Mediapipe**: A framework by Google for real-time computer vision tasks, used here to detect human pose landmarks and track people.
- **FER**: A library for detecting emotions from faces using deep learning models.

## How It Works
1. **Live Video Capture**: The system continuously captures video frames from the webcam using OpenCV.
2. **People Detection**: Using **Mediapipe Pose**, the system tracks people in the video by detecting human body landmarks. Each detected pose corresponds to one person, and their count is updated in real time.
3. **Emotion Detection**: **FER** is employed to detect the emotions of the people in the video based on their faces. The model analyzes each detected face and classifies it into different emotions (e.g., happy, sad, angry, etc.).
4. **Display**: The detected emotions and the number of people in the frame are displayed on the video in real-time using OpenCV's `cv2.putText()` function. The processed frame is then shown in the Streamlit app.

## Features
- **Real-Time Detection**: The app processes video frames continuously to detect people and emotions without significant delay.
- **Emotion Classification**: Displays the most likely emotion for each face detected in the video feed.
- **People Count**: Tracks and displays the number of people present in the video frame.
- **User Interaction**: The app provides a button to start the detection process, and users can interact with the system seamlessly.

## Installation
To run this project locally, follow the steps below.

### 1. Install Dependencies
Ensure you have Python installed, then install the necessary dependencies by running the following command:

bash
pip install streamlit opencv-python-headless mediapipe fer


Run the App
After saving the Python script (e.g., app.py), run the following command to start the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Start Detection
Click the Start Detection button in the Streamlit interface to begin live detection of people and emotions.
The webcam will start, and the app will begin processing frames in real-time, displaying both the number of people and their detected emotions.
