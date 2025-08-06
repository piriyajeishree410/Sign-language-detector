import os
import cv2
import numpy as np
import pickle
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def process_static_data(data_dir):
    """Process static signs from /data directory (JPEG images)"""
    X, y = [], []
    
    print("Processing static signs...")
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
            
        for img_file in os.listdir(label_dir):
            if img_file.endswith('.jpg'):
                img = cv2.imread(os.path.join(label_dir, img_file))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                results = mp_hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    landmarks = []
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Normalize landmarks
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    min_x, min_y = min(x_coords), min(y_coords)
                    
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x - min_x, lm.y - min_y])
                    
                    X.append(landmarks)
                    y.append(label)
        print(f"Processed {len(X)} samples for {label}")
    
    return np.array(X), np.array(y)

def process_dynamic_data(dynamic_dir):
    """Process dynamic signs from /dynamic_data directory (NPY files)"""
    X, y = [], []
    
    print("Processing dynamic signs...")
    for label in os.listdir(dynamic_dir):
        label_dir = os.path.join(dynamic_dir, label)
        if not os.path.isdir(label_dir):
            continue
            
        for seq_file in os.listdir(label_dir):
            if seq_file.endswith('.npy'):
                sequence = np.load(os.path.join(label_dir, seq_file))
                X.append(sequence)
                y.append(label)
        print(f"Processed {len(X)} sequences for {label}")
    
    return np.array(X), np.array(y)

def save_datasets(static_data, dynamic_data):
    """Save processed datasets"""
    os.makedirs('processed_data', exist_ok=True)
    
    # Save static data
    with open('processed_data/static_data.pkl', 'wb') as f:
        pickle.dump({
            'X': static_data[0],
            'y': static_data[1]
        }, f)
    
    # Save dynamic data
    with open('processed_data/dynamic_data.pkl', 'wb') as f:
        pickle.dump({
            'X': dynamic_data[0],
            'y': dynamic_data[1],
            'input_shape': (dynamic_data[0].shape[1], dynamic_data[0].shape[2])
        }, f)

if __name__ == "__main__":
    # Process both data types
    static_X, static_y = process_static_data('./data')
    dynamic_X, dynamic_y = process_dynamic_data('./dynamic_data')
    
    # Save results
    save_datasets((static_X, static_y), (dynamic_X, dynamic_y))
    
    print(f"\nSummary:")
    print(f"Static data: {static_X.shape} samples, {static_X.shape[1]} features")
    print(f"Dynamic data: {dynamic_X.shape} samples, sequences of shape {dynamic_X.shape[1:]}")