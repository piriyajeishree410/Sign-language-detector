import os
import cv2
import numpy as np
import mediapipe as mp
from utils import extract_landmarks

DATA_DIR = './dynamic_data'
SEQUENCE_LENGTH = 30  # Frames per sample

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

signs = ['thank_you', 'hello', 'goodbye']  # Sample dynamic signs
samples_per_sign = 20

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.5,
    model_complexity=1
)

for sign_idx, sign_name in enumerate(signs):
    sign_dir = os.path.join(DATA_DIR, sign_name)
    if not os.path.exists(sign_dir):
        os.makedirs(sign_dir)
    
    print(f'Collecting data for: {sign_name}')
    
    for sample_num in range(samples_per_sign):
        print(f'Sample #{sample_num+1} - Prepare to record...')
        sequence = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            cv2.putText(
                frame, f'Ready? Press "S" for {sign_name}!', 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.imshow('Collect Dynamic Data', frame)
            if cv2.waitKey(25) == ord('s'):
                break
        
        print('Recording...')
        while len(sequence) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            landmarks = extract_landmarks(frame, mp_hands)
            
            if landmarks is not None:
                sequence.append(landmarks)
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            
            # Visual feedback
            cv2.rectangle(frame, (10, 10), (250, 60), (0, 0, 0), -1)
            cv2.putText(
                frame, f'Recording: {len(sequence)}/{SEQUENCE_LENGTH}', 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
            cv2.imshow('Collect Dynamic Data', frame)
            cv2.waitKey(1)
        
        # Save sequence
        np.save(os.path.join(sign_dir, f'{sample_num}.npy'), np.array(sequence))
        print(f'Saved {sign_name} sample #{sample_num+1}')

cap.release()
cv2.destroyAllWindows()