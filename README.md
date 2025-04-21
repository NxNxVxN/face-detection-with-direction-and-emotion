# Face and Emotion Recognition System

This project combines face recognition, emotion detection, and face direction analysis in a real-time application.

## Features

- **Face Detection**: Identifies faces in real-time video
- **Person Recognition**: Recognizes known people from registered faces
- **Emotion Classification**: Detects 7 emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- **Face Direction Detection**: Determines which way a person is facing (Left, Right, Up, Down, Forward)

## Requirements

Install required packages:
```
pip install -r requirements.txt
```

## Usage

1. Run the main application:
```
python emotion_recognition.py
```

2. Register new faces:
```
python add_face.py [name]
```

3. Prepare the emotion dataset (if needed):
```
python prepare_dataset.py
```

## Controls

- Press 'R' to register a new person
- Press 'ESC' to quit

## Project Structure

- `emotion_recognition.py`: Main application with UI
- `person_recognition.py`: Face recognition functionality
- `face_direction.py`: Face direction detection
- `add_face.py`: Utility to register new faces
- `prepare_dataset.py`: Processes the FER2013 emotion dataset 