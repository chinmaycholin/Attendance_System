import face_recognition
import os
import pickle
import cv2
import numpy as np
from PIL import Image, ImageOps

dataset_path = "dataset"
processed_path = "processed"

# Create processed directory if it doesn't exist
if not os.path.exists(processed_path):
    os.makedirs(processed_path)

known_encodings = []
known_names = []

for student_name in os.listdir(dataset_path):
    student_path = os.path.join(dataset_path, student_name)
    
    if not os.path.isdir(student_path):
        continue

    # Create student folder in processed directory
    processed_student_path = os.path.join(processed_path, student_name)
    if not os.path.exists(processed_student_path):
        os.makedirs(processed_student_path)

    for img_name in os.listdir(student_path):
        img_path = os.path.join(student_path, img_name)
        
        # Skip non-image files if any
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Load image using PIL to handle EXIF rotation automatically
        try:
            pil_image = Image.open(img_path)
            pil_image = ImageOps.exif_transpose(pil_image)
            # Convert to numpy array (RGB)
            rgb_image = np.array(pil_image.convert('RGB'))
            # Create a BGR version for cropping/saving later
            image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Warning: Could not read {img_path}. Error: {e}")
            continue

        # Use a resized copy for detection to prevent CNN MemoryError
        max_size = 800
        h, w = image.shape[:2]
        scale = 1.0
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            detect_image = cv2.resize(image, (int(w * scale), int(h * scale)))
            rgb_detect = cv2.cvtColor(detect_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_detect = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face(s) using HOG (low power) with increased upsample to easily find smaller faces
        face_locations = face_recognition.face_locations(rgb_detect, model='hog', number_of_times_to_upsample=2)

        if not face_locations:
            print(f"Warning: No face found using HOG in {img_path}. Trying CNN fallback...")
            # CNN is slower but more accurate. We only run it if HOG fails.
            face_locations = face_recognition.face_locations(rgb_detect, model='cnn')

        if not face_locations:
            print(f"Warning: Still no face found in {img_path} with CNN. Skipping.")
            continue

        # Process the first face found in the image (assuming 1 person per training image)
        top, right, bottom, left = face_locations[0]

        # Scale coordinates back up to crop from the ORIGINAL high-quality image
        if scale != 1.0:
            top = int(top / scale)
            right = int(right / scale)
            bottom = int(bottom / scale)
            left = int(left / scale)

        # Crop the high-quality face
        face_crop = image[top:bottom, left:right]

        if face_crop.size == 0:
            continue
            
        # Save the full-quality processed image without shrinking it
        save_path = os.path.join(processed_student_path, img_name)
        cv2.imwrite(save_path, face_crop)
        
        # Extract encoding from the full-quality cropped RGB image
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(face_rgb)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(student_name)
            print(f"Processed and encoded: {save_path}")
        else:
            print(f"Warning: Could not extract encoding from cropped face in {img_path}.")

# Save encodings
data = {"encodings": known_encodings, "names": known_names}

with open("encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("Training complete! Processed images are saved in the 'processed/' folder.")
