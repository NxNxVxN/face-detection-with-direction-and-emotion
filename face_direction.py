import mediapipe as mp
import numpy as np
from PIL import Image

class FaceDirectionDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def get_face_direction(self, image):
        """
        Detect face direction from image
        Returns: (yaw, pitch, roll) in degrees
        """
        # Convert PIL Image to RGB numpy array
        image_array = np.array(image)
        height, width = image_array.shape[:2]
        
        # MediaPipe expects RGB
        results = self.face_mesh.process(image_array)
        
        if not results.multi_face_landmarks:
            return None
            
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get specific landmarks for pose estimation
        # Nose tip
        nose = face_landmarks.landmark[1]
        # Chin
        chin = face_landmarks.landmark[152]
        # Left eye corner
        left_eye = face_landmarks.landmark[33]
        # Right eye corner
        right_eye = face_landmarks.landmark[263]
        # Left mouth corner
        left_mouth = face_landmarks.landmark[287]
        # Right mouth corner
        right_mouth = face_landmarks.landmark[57]
        
        # Convert normalized coordinates to pixel coordinates
        nose_pos = (int(nose.x * width), int(nose.y * height))
        left_eye_pos = (int(left_eye.x * width), int(left_eye.y * height))
        right_eye_pos = (int(right_eye.x * width), int(right_eye.y * height))
        
        # Calculate face direction
        eye_center = ((left_eye_pos[0] + right_eye_pos[0]) // 2,
                     (left_eye_pos[1] + right_eye_pos[1]) // 2)
        
        # Calculate angles
        dx = right_eye_pos[0] - left_eye_pos[0]
        dy = right_eye_pos[1] - left_eye_pos[1]
        
        # Calculate roll (head tilt)
        roll = np.degrees(np.arctan2(dy, dx))
        
        # Calculate yaw (left-right rotation)
        eye_distance = np.linalg.norm(np.array(right_eye_pos) - np.array(left_eye_pos))
        reference_distance = width * 0.3  # Expected eye distance when facing forward
        yaw = np.degrees(np.arccos(min(eye_distance / reference_distance, 1.0)))
        if nose.x < 0.5:
            yaw = -yaw
            
        # Calculate pitch (up-down tilt)
        nose_to_eye_y = eye_center[1] - nose_pos[1]
        reference_y = height * 0.1  # Expected vertical distance when facing forward
        pitch = np.degrees(np.arctan2(nose_to_eye_y - reference_y, reference_y))
        
        return {
            'yaw': yaw,      # Left/Right rotation
            'pitch': pitch,  # Up/Down tilt
            'roll': roll     # Head tilt
        }
        
    def get_direction_text(self, angles):
        """Convert angles to human-readable direction"""
        if angles is None:
            return "No face detected"
            
        direction = []
        
        # Yaw (left-right)
        if angles['yaw'] < -15:
            direction.append("Looking Left")
        elif angles['yaw'] > 15:
            direction.append("Looking Right")
            
        # Pitch (up-down)
        if angles['pitch'] < -15:
            direction.append("Looking Up")
        elif angles['pitch'] > 15:
            direction.append("Looking Down")
            
        # Roll (tilt)
        if angles['roll'] < -15:
            direction.append("Head Tilted Left")
        elif angles['roll'] > 15:
            direction.append("Head Tilted Right")
            
        if not direction:
            return "Facing Forward"
            
        return ", ".join(direction) 