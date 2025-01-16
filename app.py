import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from fer import FER  # Import FER for emotion recognition

# Initialize the FER detector
detector = FER()

# Load the face cascade for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize MediaPipe Pose for tracking people
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def track_people(frame):
    """Detect and track people using MediaPipe Pose landmarks and count them."""
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Convert image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect landmarks for body pose
        results = pose.process(rgb_frame)
        
        num_people = 0  # Initialize the number of people
        
        if results.pose_landmarks:
            # Count the number of people detected based on the landmarks
            num_people += 1  # Each detected pose corresponds to one person
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        return frame, num_people

def detect_emotion(frame):
    """Detect and classify emotions using the FER library."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        
        # Use FER library to detect the emotion
        emotion, score = detector.top_emotion(face)
        
        if score is not None:
            # Draw emotion label on the face
            cv2.putText(frame, f"{emotion} ({score*100:.2f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "No Emotion Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    return frame

def process_frame(frame):
    """Process each frame for person tracking and emotion detection."""
    # Track people in the frame and count them
    frame, num_people = track_people(frame)
    
    # Detect and classify emotions
    frame = detect_emotion(frame)
    
    # Add text to display the number of people
    cv2.putText(frame, f"People Count: {num_people}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

# Streamlit Web App UI
st.title("Live People Detection and Emotion Tracking")

# Add a button to start detection
start_button = st.button('Start Detection')

# Initialize an empty container for displaying the webcam feed
frame_container = st.empty()

if start_button:
    # Open the webcam for live video
    cap = cv2.VideoCapture(0)

    # Display the live feed in the Streamlit app
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame (track people and detect emotions)
        processed_frame = process_frame(frame)
        
        # Convert frame to RGB (Streamlit requires RGB format)
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Display the processed frame in Streamlit
        frame_container.image(frame_rgb, channels="RGB", use_column_width=True)

        # You can add a condition to break the loop (e.g., stop button, time out, etc.)
        # Here it's just running continuously. You can add break logic for better control.

    cap.release()
else:
    st.write("Click the button to start detecting people and emotions!")
