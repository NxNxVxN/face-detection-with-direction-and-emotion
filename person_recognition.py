import os
import pickle
import numpy as np
import face_recognition
from PIL import Image

class PersonRecognizer:
    def __init__(self):
        # Directory to store person data
        self.data_dir = 'person_data'
        os.makedirs(self.data_dir, exist_ok=True)
        
        # File to store face encodings and names
        self.data_file = os.path.join(self.data_dir, 'people.pkl')
        
        # Load existing data if available
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_data()
    
    def load_data(self):
        """Load known face encodings and names from file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"Loaded {len(self.known_face_names)} people: {', '.join(self.known_face_names)}")
            except Exception as e:
                print(f"Error loading person data: {e}")
                # Initialize empty lists if loading fails
                self.known_face_encodings = []
                self.known_face_names = []
        else:
            print("No existing person data found.")
    
    def save_data(self):
        """Save known face encodings and names to file"""
        try:
            with open(self.data_file, 'wb') as f:
                data = {
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }
                pickle.dump(data, f)
            print(f"Saved {len(self.known_face_names)} people to {self.data_file}")
        except Exception as e:
            print(f"Error saving person data: {e}")
    
    def register_person(self, face_image, name):
        """Register a new person with their face image and name"""
        try:
            # Convert PIL Image to RGB
            if face_image.mode != 'RGB':
                face_image = face_image.convert('RGB')
            
            # Convert to numpy array
            face_array = np.array(face_image)
            
            # Find face encodings
            face_encodings = face_recognition.face_encodings(face_array)
            
            if not face_encodings:
                print("No face found in the image.")
                return False
            
            # Use the first face found
            face_encoding = face_encodings[0]
            
            # Check if this person is already registered
            if name in self.known_face_names:
                idx = self.known_face_names.index(name)
                # Update encoding
                self.known_face_encodings[idx] = face_encoding
                print(f"Updated face encoding for {name}")
            else:
                # Add new person
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
                print(f"Registered new person: {name}")
            
            # Save the updated data
            self.save_data()
            return True
        
        except Exception as e:
            print(f"Error registering person: {e}")
            return False
    
    def recognize_person(self, face_image):
        """Recognize a person from their face image"""
        try:
            # Convert PIL Image to RGB
            if face_image.mode != 'RGB':
                face_image = face_image.convert('RGB')
            
            # Convert to numpy array
            face_array = np.array(face_image)
            
            # Find face encodings
            face_encodings = face_recognition.face_encodings(face_array)
            
            if not face_encodings:
                return None, 0.0
            
            # Use the first face found
            face_encoding = face_encodings[0]
            
            if not self.known_face_encodings:
                return None, 0.0
            
            # Compare face with known faces
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            # Calculate confidence (lower distance = higher confidence)
            confidence = 1.0 - min(face_distances[best_match_index], 1.0)
            
            # Return name and confidence if confidence is high enough
            if confidence > 0.5:
                return self.known_face_names[best_match_index], confidence
            else:
                return None, confidence
        
        except Exception as e:
            print(f"Error recognizing person: {e}")
            return None, 0.0
    
    def get_registered_people(self):
        """Get a list of registered people"""
        return self.known_face_names 