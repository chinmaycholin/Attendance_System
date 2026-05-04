import cv2
import face_recognition
import numpy as np
import os

# Load the OpenCV DNN Face Detector
PROTOTXT_PATH = os.path.join("packages", "deploy.prototxt")
MODEL_PATH = os.path.join("packages", "res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(MODEL_PATH):
    print("❌ ERROR: OpenCV DNN models not found! Please run 'python download_models.py' first.")
    net = None
else:
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

def recognize_faces(frame, known_encodings, known_names):
    if net is None:
        return []

    (h, w) = frame.shape[:2]
    
    # --- STAGE 1: Fast Candidate Detection using OpenCV DNN ---
    # The SSD model uses mathematically fixed anchor boxes trained precisely on 300x300 inputs.
    # If we pass the full 1080p resolution, the math breaks and it fails to see faces!
    # (Don't worry, the image is shrunk ONLY for finding the coordinates. The actual face crop later uses the full HD image.)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    results = []
    
    # --- STAGE 2: Padded Cropping & Identification ---
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Lowered threshold to 0.3 to help detect side-profile faces (which typically have lower confidence)
        if confidence > 0.3:
            # Map coordinates back to the original full-res image
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Add 20% padding so we don't cut off chin/forehead during encoding
            pad_w = int((endX - startX) * 0.2)
            pad_h = int((endY - startY) * 0.2)
            
            # Ensure coordinates are within image boundaries
            startX = max(0, startX - pad_w)
            startY = max(0, startY - pad_h)
            endX = min(w, endX + pad_w)
            endY = min(h, endY + pad_h)
            
            # Crop the padded face from the FULL-RESOLUTION image
            face_crop = frame[startY:endY, startX:endX]
            
            if face_crop.size == 0:
                continue
                
            # Convert to RGB for face_recognition
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            crop_h, crop_w = face_rgb.shape[:2]
            
            # --- STAGE 3: CNN Identification ---
            # By explicitly providing the bounding box, we tell face_recognition 
            # to skip its slow internal face detection entirely and jump straight to CNN encoding!
            face_bounding_box = [(0, crop_w, crop_h, 0)] # (top, right, bottom, left)
            
            encodings = face_recognition.face_encodings(face_rgb, known_face_locations=face_bounding_box)
            
            if len(encodings) > 0:
                enc = encodings[0]
                distances = face_recognition.face_distance(known_encodings, enc)
                min_dist = np.min(distances)
                
                if min_dist < 0.45:
                    name = known_names[np.argmin(distances)]
                    color = (0, 255, 0)
                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                    
                # Append formatted results for the draw_boxes function: (top, right, bottom, left, name, color)
                results.append((startY, endX, endY, startX, name, color))

    return results

def draw_boxes(frame, results):
    for (top, right, bottom, left, name, color) in results:
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom-25), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left+5, bottom-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    return frame
