# ✋🤟 Sign Language to Text Conversion System

A **two-level real-time sign language recognition system** that supports both **static gestures (alphabets)** and **dynamic gestures (words/phrases)**.  

Built using **OpenCV, MediaPipe, scikit-learn, and TensorFlow**.

---

## System Architecture
The project is divided into **two levels**:

### 1️) Static Gesture Recognition
- **Model**: Random Forest Classifier (scikit-learn)  
- **Input**: Extracted hand landmarks (relative positions, pinch distance, finger spread)  
- **Output**: Alphabets (A–Z) and control gestures (SPACE, DELETE, MODE)  
- **Accuracy**: **99.81%**

### 2️) Dynamic Gesture Recognition
- **Model**: LSTM (Long Short-Term Memory) Neural Network (TensorFlow)  
- **Input**: Sequences of hand landmark frames (temporal data)  
- **Output**: Words and short phrases  
- **Accuracy**: **100%**

---

## Features
- Dual architecture: **Random Forest (static)** + **LSTM (dynamic)**.  
- Real-time hand tracking with **MediaPipe** + **OpenCV**.  
- Robust accuracy (**99.81% static, 100% dynamic**).   
- Designed for **accessibility & inclusivity**.  

---

## Tech Stack
- **Computer Vision**: OpenCV, MediaPipe  
- **Machine Learning**: scikit-learn (Random Forest)  
- **Deep Learning**: TensorFlow (LSTM)

---

## Demo
[I will add screenshots here shortly]

---

## Usage
1. Clone repo:  
   ```bash
   git clone https://github.com/piriyajeishree410/Sign-language-detector.git
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the system:

   ```bash
   python main.py
   ```

---

## Future Improvements

* Expand dynamic dataset to support full sentences.
* Add multilingual text-to-speech output.
* Deploy as a **web or mobile app** for broader accessibility.

---

```

---
