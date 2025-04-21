import os
import numpy as np
import pygame
import pygame.camera
import mediapipe as mp
from PIL import Image
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from person_recognition import PersonRecognizer
from face_direction import FaceDirectionDetector

class EmotionRecognizer:
    def __init__(self):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for close-range, 1 for far-range
            min_detection_confidence=0.5
        )
        
        # Initialize face direction detector
        self.face_direction_detector = FaceDirectionDetector()
        
        # Initialize emotion recognition model
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.model = None
        self.scaler = None
        self.load_emotion_model()
        
        # Initialize person recognizer
        self.person_recognizer = PersonRecognizer()
        
        # Mode flags
        self.registration_mode = False
        self.current_registration_name = ""
        
        # Initialize pygame for webcam
        pygame.init()
        pygame.camera.init()
        
        # Find available cameras
        cameras = pygame.camera.list_cameras()
        if not cameras:
            raise ValueError("No cameras found")
        
        # Initialize camera
        self.camera_size = (640, 480)
        self.camera = pygame.camera.Camera(cameras[0], self.camera_size)
        self.camera.start()
        
        # Set up display
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Emotion and Person Recognition")
        
        # Initialize clock for FPS control
        self.clock = pygame.time.Clock()
        
        # Initialize font
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
    
    def create_neural_network(self, input_shape=(48, 48, 1), num_classes=7):
        """Create a convolutional neural network for emotion recognition"""
        model = Sequential()
        
        # First convolutional layer
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Second convolutional layer
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        
        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        
        return model
        
    def load_emotion_model(self):
        """Load pre-trained model and scaler or train a new one"""
        try:
            # Load the scaler
            scaler_path = 'data/scaler.joblib'
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler not found at {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            
            # Load training data
            X_train = np.load('data/X_train.npy')
            y_train = np.load('data/y_train.npy')
            
            # Reshape data for CNN (expecting 48x48x1 grayscale images)
            # First, scale the data back to 0-255 range if it's not already
            X_train_unscaled = self.scaler.inverse_transform(X_train)
            
            # Reshape to 48x48 and add channel dimension
            num_samples = X_train_unscaled.shape[0]
            X_train_reshaped = X_train_unscaled.reshape(num_samples, 48, 48, 1)
            
            # Normalize to 0-1 range for neural network
            X_train_normalized = X_train_reshaped / 255.0
            
            # Convert labels to one-hot encoding
            y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=7)
            
            # Always train a new model due to compatibility issues with saved models
            print("Training new neural network model...")
            self.model = self.create_neural_network()
            self.model.fit(
                X_train_normalized, 
                y_train_onehot,
                batch_size=64,
                epochs=5,  # Limited epochs for demonstration
                validation_split=0.2
            )
            # Save the model
            os.makedirs('data', exist_ok=True)
            model_path = 'data/emotion_model.keras'
            self.model.save(model_path)
                
            print("Model training completed!")
            
        except Exception as e:
            print(f"Error loading/training model: {e}")
            raise
    
    def preprocess_face(self, face_img):
        """Preprocess face image for emotion recognition"""
        try:
            # Convert to grayscale
            if face_img.mode != 'L':
                face_img = face_img.convert('L')
                
            # Resize to 48x48
            face_img = face_img.resize((48, 48))
            
            # Convert to numpy array
            face_array = np.array(face_img)
            
            # Normalize pixel values to 0-1 for neural network
            face_normalized = face_array / 255.0
            
            # Add batch and channel dimensions
            face_tensor = face_normalized.reshape(1, 48, 48, 1)
            
            return face_tensor
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None
    
    def detect_emotion(self, face_img):
        """Detect emotion in a face image"""
        try:
            # Preprocess the face
            face_tensor = self.preprocess_face(face_img)
            if face_tensor is None:
                return None, 0.0
            
            # Predict emotion
            prediction = self.model.predict(face_tensor)[0]
            emotion_idx = np.argmax(prediction)
            confidence = float(prediction[emotion_idx])
            
            return self.emotions[emotion_idx], confidence
        except Exception as e:
            print(f"Error detecting emotion: {e}")
            return None, 0.0
    
    def draw_results(self, surface, face_locations, emotions, person_names):
        """Draw face boxes, emotion labels, and person names on the surface"""
        try:
            # Clear the screen
            self.screen.fill((0, 0, 0))
            
            # Draw face boxes and emotions
            for (x, y, w, h), (emotion, emotion_conf), (person, person_conf) in zip(face_locations, emotions, person_names):
                if emotion is not None:
                    # Draw face box
                    pygame.draw.rect(surface, (0, 255, 0), (x, y, w, h), 2)
                    
                    # Get face direction
                    # Convert pygame surface to numpy array properly
                    surface_string = pygame.image.tostring(surface, 'RGB')
                    surface_pil = Image.frombytes('RGB', surface.get_size(), surface_string)
                    surface_array = np.array(surface_pil)
                    
                    # Now we can safely slice the array
                    if y >= 0 and x >= 0 and y+h <= surface_array.shape[0] and x+w <= surface_array.shape[1]:
                        face_img = Image.fromarray(surface_array[y:y+h, x:x+w])
                        direction_angles = self.face_direction_detector.get_face_direction(face_img)
                        direction_text = self.face_direction_detector.get_direction_text(direction_angles)
                        
                        # Draw face direction
                        direction_label = self.small_font.render(direction_text, True, (0, 255, 255))
                        surface.blit(direction_label, (x, y - 100))
                    
                    # Draw emotion label
                    emotion_label = f"{emotion} ({emotion_conf:.2f})"
                    emotion_text = self.font.render(emotion_label, True, (0, 255, 0))
                    surface.blit(emotion_text, (x, y - 40))
                    
                    # Draw person name if recognized
                    if person is not None:
                        person_label = f"{person} ({person_conf:.2f})"
                        person_text = self.font.render(person_label, True, (255, 255, 0))
                        surface.blit(person_text, (x, y - 70))
                    else:
                        person_text = self.font.render("Unknown Person", True, (255, 0, 0))
                        surface.blit(person_text, (x, y - 70))
            
            # Draw the image on the screen
            self.screen.blit(pygame.transform.scale(surface, (self.width, self.height)), (0, 0))
            
            # Draw registration mode status
            if self.registration_mode:
                status_text = self.small_font.render(f"Registration Mode: Enter name for the face", True, (255, 255, 255))
                self.screen.blit(status_text, (10, 10))
                
                name_text = self.font.render(f"Name: {self.current_registration_name}_", True, (255, 255, 255))
                self.screen.blit(name_text, (10, 40))
            else:
                # Draw instructions
                instructions = [
                    "Press 'R' to register a new person",
                    "Press 'ESC' to quit"
                ]
                for i, instruction in enumerate(instructions):
                    text = self.small_font.render(instruction, True, (255, 255, 255))
                    self.screen.blit(text, (10, 10 + i * 25))
                
                # Display registered people
                people = self.person_recognizer.get_registered_people()
                if people:
                    text = self.small_font.render(f"Registered People: {', '.join(people)}", True, (255, 255, 255))
                    self.screen.blit(text, (10, self.height - 30))
            
            # Update display
            pygame.display.flip()
        except Exception as e:
            print(f"Error drawing results: {e}")
    
    def run(self):
        """Main loop for real-time emotion and person recognition"""
        try:
            running = True
            while running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_r and not self.registration_mode:
                            # Enter registration mode
                            self.registration_mode = True
                            self.current_registration_name = ""
                        elif self.registration_mode:
                            if event.key == pygame.K_RETURN and self.current_registration_name:
                                # Register the current face with the entered name
                                frame = self.camera.get_image()
                                frame_string = pygame.image.tostring(frame, 'RGB')
                                pil_image = Image.frombytes('RGB', self.camera_size, frame_string)
                                
                                results = self.face_detection.process(np.array(pil_image))
                                if results.detections:
                                    # Use the first detected face
                                    detection = results.detections[0]
                                    bbox = detection.location_data.relative_bounding_box
                                    h, w, _ = np.array(pil_image).shape
                                    x = int(bbox.xmin * w)
                                    y = int(bbox.ymin * h)
                                    width = int(bbox.width * w)
                                    height = int(bbox.height * h)
                                    
                                    face_region = pil_image.crop((x, y, x + width, y + height))
                                    success = self.person_recognizer.register_person(face_region, self.current_registration_name)
                                    
                                    if success:
                                        print(f"Successfully registered {self.current_registration_name}")
                                    else:
                                        print(f"Failed to register {self.current_registration_name}")
                                
                                # Exit registration mode
                                self.registration_mode = False
                                self.current_registration_name = ""
                            elif event.key == pygame.K_BACKSPACE:
                                # Handle backspace in name entry
                                self.current_registration_name = self.current_registration_name[:-1]
                            elif event.unicode.isalnum() or event.unicode.isspace():
                                # Add character to name
                                self.current_registration_name += event.unicode
                
                # Capture frame using pygame
                frame = self.camera.get_image()
                
                # Convert pygame surface to PIL Image for MediaPipe
                frame_string = pygame.image.tostring(frame, 'RGB')
                pil_image = Image.frombytes('RGB', self.camera_size, frame_string)
                frame_array = np.array(pil_image)
                
                # Detect faces using MediaPipe
                results = self.face_detection.process(frame_array)
                
                face_locations = []
                emotions = []
                person_names = []
                
                if results.detections and not self.registration_mode:
                    for detection in results.detections:
                        # Get face bounding box
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = frame_array.shape
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # Extract face region
                        face_region = pil_image.crop((x, y, x + width, y + height))
                        
                        # Detect emotion
                        emotion, emotion_conf = self.detect_emotion(face_region)
                        
                        # Recognize person
                        person, person_conf = self.person_recognizer.recognize_person(face_region)
                        
                        face_locations.append((x, y, width, height))
                        emotions.append((emotion, emotion_conf))
                        person_names.append((person, person_conf))
                
                # Draw results
                self.draw_results(frame, face_locations, emotions, person_names)
                
                # Control FPS
                self.clock.tick(30)
                
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.camera.stop()
            pygame.quit()
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    recognizer = EmotionRecognizer()
    recognizer.run() 