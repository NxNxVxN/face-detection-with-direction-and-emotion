import os
import pygame
import pygame.camera
from PIL import Image
import face_recognition
import time
import sys

def capture_face(name):
    """Capture a face image and save it to the known_faces directory with the given name"""
    # Initialize pygame
    pygame.init()
    pygame.camera.init()
    
    # Find available cameras
    cameras = pygame.camera.list_cameras()
    if not cameras:
        print("No cameras found!")
        return False
    
    # Initialize the first camera
    camera_size = (640, 480)
    camera = pygame.camera.Camera(cameras[0], camera_size)
    camera.start()
    
    # Set up display
    screen = pygame.display.set_mode(camera_size)
    pygame.display.set_caption("Face Capture")
    
    # Initialize font for text
    font = pygame.font.Font(None, 36)
    
    # Ensure the known_faces directory exists
    os.makedirs("known_faces", exist_ok=True)
    
    print(f"Capturing face for {name}...")
    print("Position your face in front of the camera")
    print("Press SPACE to capture, ESC to cancel")
    
    # Main loop
    running = True
    captured = False
    countdown = 0
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and countdown == 0:
                    # Start countdown
                    countdown = 3
        
        # Get frame from camera
        frame = camera.get_image()
        
        # Display the camera feed
        screen.blit(frame, (0, 0))
        
        if countdown > 0:
            # Display countdown
            text = font.render(f"Capturing in {countdown}...", True, (255, 0, 0))
            screen.blit(text, (20, 20))
            pygame.display.flip()
            
            # Wait for 1 second
            time.sleep(1)
            countdown -= 1
            
            # Capture and save when countdown reaches 0
            if countdown == 0:
                # Get image data
                frame_string = pygame.image.tostring(frame, 'RGB')
                pil_image = Image.frombytes('RGB', camera_size, frame_string)
                
                # Save the image
                filename = os.path.join("known_faces", f"{name}.jpg")
                pil_image.save(filename)
                
                # Verify that a face is detected
                image = face_recognition.load_image_file(filename)
                face_locations = face_recognition.face_locations(image)
                
                if face_locations:
                    print(f"Face captured successfully! Saved as {filename}")
                    captured = True
                    running = False
                else:
                    print("No face detected in the image. Please try again.")
                    os.remove(filename)  # Remove the file with no face
                    countdown = 0
        else:
            # Display instructions
            text = font.render("Press SPACE to capture", True, (0, 255, 0))
            screen.blit(text, (20, 20))
        
        # Update display
        pygame.display.flip()
    
    # Clean up
    camera.stop()
    pygame.quit()
    
    return captured

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_face.py <name>")
        print("Example: python add_face.py John")
        sys.exit(1)
    
    name = sys.argv[1]
    capture_face(name) 