import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image
import glob

def load_images_from_directory(directory):
    images = []
    labels = []
    emotion_map = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'neutral': 4,
        'sad': 5,
        'surprise': 6
    }
    
    # Get the emotion from the directory name
    emotion = os.path.basename(directory)
    if emotion not in emotion_map:
        return [], []
    
    # Load all images in the directory
    image_files = glob.glob(os.path.join(directory, '*.jpg'))
    for image_file in image_files:
        try:
            # Open and convert image to grayscale
            img = Image.open(image_file).convert('L')
            # Resize to 48x48
            img = img.resize((48, 48))
            # Convert to numpy array
            img_array = np.array(img)
            images.append(img_array)
            labels.append(emotion_map[emotion])
        except Exception as e:
            print(f"Error loading {image_file}: {e}")
    
    return np.array(images), np.array(labels)

def prepare_fer2013():
    print("Downloading FER2013 dataset...")
    import kagglehub
    dataset_path = kagglehub.dataset_download("msambare/fer2013")
    print("Dataset downloaded to:", dataset_path)
    
    # Define paths for train and test data
    base_path = os.path.join("kagglehub", "datasets", "msambare", "fer2013", "versions", "1")
    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "test")
    
    print("Loading training data from:", train_path)
    print("Loading test data from:", test_path)
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Dataset directories not found")
    
    # Process training data
    print("Processing training data...")
    X_train = []
    y_train = []
    
    # Process each emotion directory in train
    for emotion_dir in os.listdir(train_path):
        emotion_path = os.path.join(train_path, emotion_dir)
        if os.path.isdir(emotion_path):
            print(f"Processing {emotion_dir}...")
            X, y = load_images_from_directory(emotion_path)
            X_train.extend(X)
            y_train.extend(y)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Process test data
    print("Processing test data...")
    X_test = []
    y_test = []
    
    # Process each emotion directory in test
    for emotion_dir in os.listdir(test_path):
        emotion_path = os.path.join(test_path, emotion_dir)
        if os.path.isdir(emotion_path):
            print(f"Processing {emotion_dir}...")
            X, y = load_images_from_directory(emotion_path)
            X_test.extend(X)
            y_test.extend(y)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Loaded {len(X_train)} training images and {len(X_test)} test images")
    
    # Flatten images for RandomForest
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Save the processed data
    print("Saving processed data...")
    os.makedirs('data', exist_ok=True)
    np.save('data/X_train.npy', X_train_scaled)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_test.npy', X_test_scaled)
    np.save('data/y_test.npy', y_test)
    joblib.dump(scaler, 'data/scaler.joblib')
    
    print("Dataset preparation completed!")
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler

if __name__ == "__main__":
    prepare_fer2013() 