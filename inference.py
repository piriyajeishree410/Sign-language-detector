import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.models import load_model
from utils import extract_landmarks

# Configuration
STATIC_MODEL_PATH = 'models/static_model.p'
DYNAMIC_MODEL_PATH = 'models/dynamic_model.h5'
SEQUENCE_LENGTH = 30
MIN_DETECTION_THRESHOLD = 0.6  # 60% frames need hands

# Load models
static_model = pickle.load(open(STATIC_MODEL_PATH, 'rb'))['model']
dynamic_model = load_model(DYNAMIC_MODEL_PATH)
label_map = np.load('models/label_map.npy', allow_pickle=True)
static_labels = {0: 'A', 1: 'B', 2: 'L'}

# Initialize MediaPipe
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

sequence = []
current_mode = "static"  # static/dynamic
predictions = []
sentence = []

while True:
    ret, frame = cap.read()
    landmarks = extract_landmarks(frame, mp_hands)
    
    # Mode selection UI
    cv2.putText(
        frame, f"MODE: {current_mode.upper()}", 
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )
    cv2.putText(
        frame, "Press 'S': Static  'D': Dynamic  'C': Clear", 
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1
    )
    
    if landmarks is not None:
        # Static sign detection
        if current_mode == "static":
            prediction = static_model.predict([landmarks])[0]
            char = static_labels[int(prediction)]
            cv2.putText(
                frame, f"Sign: {char}", 
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
        
        # Dynamic sign detection
        else:
            sequence.append(landmarks)
            if len(sequence) > SEQUENCE_LENGTH:
                sequence = sequence[-SEQUENCE_LENGTH:]
                
                # Predict when sufficient frames have hands
                if len(sequence) == SEQUENCE_LENGTH:
                    valid_frames = sum(1 for lm in sequence if lm is not None)
                    if valid_frames / SEQUENCE_LENGTH > MIN_DETECTION_THRESHOLD:
                        # Prepare input
                        input_seq = np.array(sequence).reshape(
                            1, SEQUENCE_LENGTH, -1
                        )
                        prediction = dynamic_model.predict(input_seq)[0]
                        sign_idx = np.argmax(prediction)
                        
                        # Update sentence if high confidence
                        if prediction[sign_idx] > 0.8:
                            current_sign = label_map[sign_idx]
                            predictions.append(current_sign)
                            
                            # Detect sign end (repeated predictions)
                            if len(predictions) > 5:
                                most_common = max(
                                    set(predictions[-5:]), 
                                    key=predictions.count
                                )
                                if most_common != sentence[-1] if sentence else True:
                                    sentence.append(most_common)
                                    sequence = []  # Reset for next sign
            
            # Display dynamic sign info
            cv2.putText(
                frame, f"Recording: {len(sequence)}/{SEQUENCE_LENGTH}", 
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            if sentence:
                cv2.putText(
                    frame, f"Sentence: {' '.join(sentence)}", 
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
                )
    
    # Key controls
    key = cv2.waitKey(1)
    if key == ord('s'):
        current_mode = "static"
        sequence = []
        sentence = []
    elif key == ord('d'):
        current_mode = "dynamic"
    elif key == ord('c'):
        sentence = []
    
    cv2.imshow('Sign Language Translator', frame)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()