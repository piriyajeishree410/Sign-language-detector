import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue
        
    for img_file in os.listdir(label_dir):
        if img_file.endswith('.jpg'):
            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []
                
                hand_landmarks = results.multi_hand_landmarks[0]
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)
                
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))
                
                data.append(data_aux)
                labels.append(label)

# Save dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Created dataset with {len(data)} samples")
print("Saved to data.pickle")