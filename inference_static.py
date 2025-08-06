import os
import pickle
import cv2
import numpy as np
import mediapipe as mp

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Constants
MODEL_PATH = 'models/static_model.p'  # Standardized path

# Load model
try:
    model_dict = pickle.load(open(MODEL_PATH, 'rb'))
    model = model_dict['model']
    print("Static sign model loaded successfully")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please run train_classifier.py first to generate the model")
    exit(1)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label mapping (update this with your actual classes)
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit(1)

print("Starting sign language detection...")
print("Press ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Hand landmarks detection
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Collect and normalize landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_char = labels_dict[int(prediction[0])]

        # Bounding box
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Visualization
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(frame, predicted_char, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Static Sign Detection', frame)
    
    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()