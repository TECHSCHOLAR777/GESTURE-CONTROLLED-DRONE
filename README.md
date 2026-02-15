🚁 Gesture Controlled Drone – AI Based Control System

This project explores building an intelligent drone control system using hand gestures, machine learning, and real-time voice commands. Instead of relying on fixed rule-based gestures, the system uses MediaPipe to extract hand landmarks and trains neural networks (ANN and CNN) to recognize gestures accurately in real time.

To make control smoother and more natural, the system combines gesture prediction with temporal smoothing, gesture sequences for quick safety actions, offline low-latency voice commands using Vosk, and smile-based speed control.

The goal is to create a responsive and intuitive human-machine interface for drone navigation.

✋ What This System Can Do

Detect hand gestures in real time using MediaPipe

Classify gestures using both ANN and CNN models

Smooth predictions to avoid jitter and false triggers

Trigger emergency and boost actions using gesture sequences

Control the drone using fast offline voice commands

Adjust speed dynamically based on facial smile detection

Run fully in real time with a modular design

🧠 Gestures Used

Open Palm
Fist
Index Up
Index Down
Two Fingers Up
Three Fingers Up
“I Am Cool” gesture

Each gesture maps to a specific drone action or movement.

📈 Model Training

Both ANN and CNN models are trained on MediaPipe hand landmark data collected in real time.
The CNN model shows strong learning behavior with improving accuracy and stable validation loss across 7 gesture classes.

📂 Project Structure
camera.py
hand_landmarks.py
record_dataset.py
train_ann.py
predictor.py
smoother.py
speed_control.py
voice.py
main.py
gesture_map.json
requirements.txt
.gitignore

🚀 Getting Started
Create virtual environment
python -m venv venv
venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Download Vosk voice model

Get vosk-model-small-en-us-0.15 from:
https://alphacephei.com/vosk/models

Extract it inside the project folder.

📸 Record Gesture Data
python record_dataset.py


This saves landmark data into a CSV file for training.

🧠 Train the Model
python train_ann.py


This generates the trained gesture model.

⚠ Tips for Better Results

Keep lighting consistent

Hold hand at similar distance from camera

Record more samples for each gesture

Avoid fast or shaky movements during recording

🛠 Technologies Used

Python • MediaPipe • PyTorch • OpenCV • Vosk • NumPy • Scikit-learn

🎮 Run the System
python main.py
