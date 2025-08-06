import os
import cv2
import numpy as np
import mediapipe as mp

def extract_landmarks(frame, hands_model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(frame_rgb)
    landmarks = []
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract and normalize landmarks
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        min_x, min_y = min(x_coords), min(y_coords)
        
        for lm in hand_landmarks.landmark:
            landmarks.append(lm.x - min_x)
            landmarks.append(lm.y - min_y)
            
    return np.array(landmarks) if landmarks else None

def create_sequences(dataset_path, sequence_length):
    sequences = []
    labels = []
    sign_names = sorted(os.listdir(dataset_path))
    
    for label_idx, sign_name in enumerate(sign_names):
        sign_dir = os.path.join(dataset_path, sign_name)
        
        for seq_file in os.listdir(sign_dir):
            sequence = np.load(os.path.join(sign_dir, seq_file))
            
            # Pad or trim sequences to fixed length
            if len(sequence) > sequence_length:
                sequence = sequence[:sequence_length]
            elif len(sequence) < sequence_length:
                padding = np.zeros((sequence_length - len(sequence), sequence.shape[1]))
                sequence = np.vstack((sequence, padding))
                
            sequences.append(sequence)
            labels.append(label_idx)
    
    return np.array(sequences), np.array(labels), sign_names