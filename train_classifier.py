import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from utils import create_sequences

# Configuration
SEQUENCE_LENGTH = 30
LANDMARK_COUNT = 42  # 21 landmarks * (x,y)

# Load and preprocess data
X, y, label_map = create_sequences('./dynamic_data', SEQUENCE_LENGTH)
y = to_categorical(y)  # One-hot encode labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# LSTM Model
model = Sequential([
    LSTM(128, input_shape=(SEQUENCE_LENGTH, LANDMARK_COUNT), return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(label_map), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    batch_size=16
)

# Save model
model.save('models/dynamic_model.h5')
np.save('models/label_map.npy', label_map)
print(f"Model saved. Accuracy: {model.evaluate(X_test, y_test)[1]*100:.2f}%")