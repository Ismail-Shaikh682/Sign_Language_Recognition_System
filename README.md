# Sign_Language_Recognition_System
This is an ML based sign language recognition system which converts hand gestures into readable text format in both American as well as the Indian Sign Language
# 🤟 Sign Language Recognition System

A real-time **Sign Language Recognition System** built using **Python, OpenCV, MediaPipe, Scikit-learn, and Flask** that recognizes hand gestures through a webcam and converts them into readable text and speech. The project aims to bridge the communication gap between hearing-impaired individuals and non-sign language users by providing an efficient and user-friendly gesture recognition system.

Unlike traditional image-based deep learning approaches, this project uses **MediaPipe Hands** to extract 21 hand landmarks from each gesture and trains a **Random Forest Classifier** on normalized landmark coordinates. This approach provides faster training, requires a smaller dataset, and enables accurate real-time prediction even on systems without a dedicated GPU.

---

# 🚀 Features

- 🔴 Real-time hand gesture recognition using webcam
- ✋ Hand landmark detection using Google MediaPipe
- 🤖 Machine Learning-based gesture classification using Random Forest
- 🔤 English Alphabet Recognition (A–Z)
- 🇮🇳 Hindi Vowel Recognition
- 🇮🇳 Hindi Consonant Recognition
- ⌨️ Special Commands
  - Space
  - Backspace
  - Speak
- 🔊 Offline Text-to-Speech using pyttsx3
- 📈 Stable prediction mechanism to reduce false detections
- ⏱️ Cooldown timer to prevent duplicate character insertion
- 📦 Automatic dataset generation from captured images
- 💾 Model training and persistence using Scikit-learn
- 🖥️ Simple and interactive graphical interface

---

# 📌 Problem Statement

Communication between hearing-impaired individuals and non-signers often requires a human interpreter. This project provides a low-cost AI-based solution that recognizes sign language gestures in real time and converts them into readable text and speech, making communication more accessible and efficient.

---

# 💡 Solution

The system captures live video through a webcam and detects the user's hand using MediaPipe. Instead of processing the entire image, it extracts 21 hand landmarks and converts them into normalized numerical features. These features are then classified using a trained Random Forest model. The recognized gesture is converted into text and displayed on the screen. Users can also hear the generated sentence through offline text-to-speech functionality.

---

# ⚙️ Project Workflow

```
Webcam
   │
   ▼
OpenCV Video Capture
   │
   ▼
MediaPipe Hand Detection
   │
   ▼
21 Hand Landmarks
   │
   ▼
Landmark Normalization
   │
   ▼
42 Feature Vector
   │
   ▼
Random Forest Classifier
   │
   ▼
Gesture Prediction
   │
   ▼
Sentence Formation
   │
   ▼
Text-to-Speech
```

---

# 🧠 Machine Learning Pipeline

### Step 1 – Data Collection

- Capture gesture images using webcam
- Images are stored category-wise
- Approximately **150 images per gesture** are collected

---

### Step 2 – Feature Extraction

Each image is processed using **MediaPipe Hands**, which detects **21 hand landmarks**.

Each landmark provides:

- X coordinate
- Y coordinate

Total Features:

```
21 Landmarks × 2 Coordinates = 42 Features
```

The coordinates are normalized using the wrist position and hand size to improve prediction accuracy regardless of camera distance or hand size.

---

### Step 3 – Dataset Creation

The extracted landmark coordinates are stored in:

- dataset.csv

Each row contains:

```
42 Features + Label
```

---

### Step 4 – Model Training

The dataset is divided into training and testing sets.

The project uses a **Random Forest Classifier** with:

- 300 Decision Trees
- Balanced Class Weights
- Stratified Train-Test Split
- 5-Fold Cross Validation

The trained model is saved as:

```
model.pkl
```

---

### Step 5 – Real-Time Prediction

During live prediction:

- Webcam captures the hand
- MediaPipe extracts landmarks
- Landmarks are normalized
- Random Forest predicts the gesture
- Stable predictions are added to the sentence
- Text is converted into speech using pyttsx3

---

# 📂 Project Structure

```
Sign-Language-Recognition/
│
├── dataset/
│
├── 1_collect_data.py
├── 2_create_dataset.py
├── 3_train_model.py
├── 4_predict.py
├── main.py
│
├── dataset.csv
├── model.pkl
├── label_map.json
├── model_labels.json
│
├── templates/
├── static/
│
└── README.md
```

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Computer Vision

- OpenCV

### Hand Tracking

- Google MediaPipe Hands

### Machine Learning

- Scikit-learn
- Random Forest Classifier

### Data Processing

- NumPy
- Pandas

### Image Processing

- Pillow (PIL)

### Text-to-Speech

- pyttsx3

### Backend

- Flask

---

# 📚 Libraries Used

- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Pillow
- pyttsx3
- Pickle
- JSON

---

# ✨ Key Features Implemented

✔ Landmark Normalization

✔ Stable Gesture Recognition

✔ Prediction Cooldown

✔ Real-Time Webcam Detection

✔ English + Hindi Gesture Support

✔ Offline Speech Generation

✔ Automatic Dataset Creation

✔ Model Persistence

✔ Lightweight Machine Learning Pipeline

---

# 🎯 Future Improvements

- Continuous Sign Language Sentence Recognition
- Deep Learning-based CNN/LSTM Models
- Transformer-based Gesture Recognition
- Mobile Application Support
- Cloud Deployment
- Indian Sign Language (ISL) Vocabulary Expansion
- Voice Command Integration
- Multi-Hand Gesture Recognition

---

# 👨‍💻 Author

**Ismail Shaikh**

B.E. Information Technology

Mumbai, India

GitHub: https://github.com/Ismail-Shaikh682

LinkedIn: https://linkedin.com/in/ismail-shaikh-b769b9363
