import cv2
import pickle
import time
from collections import defaultdict
import pandas as pd
import threading
import urllib.request
import numpy as np
from recognize import recognize_faces, draw_boxes

# Load encodings
try:
    with open("encodings.pkl", "rb") as f:
        data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]
except FileNotFoundError:
    print("Error: encodings.pkl not found. Please run train.py first.")
    exit(1)

class VideoCaptureThread:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # This background thread constantly reads the MJPEG stream
        # so OpenCV never falls behind and drops the connection!
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# Camera Configuration
base_url = "http://10.188.86.143:8080"
video_url = f"{base_url}/video"
snapshot_url = f"{base_url}/shot.jpg"

cap = VideoCaptureThread(video_url)

attendance_counter = defaultdict(int)

print("Starting Attendance System...")
print("---------------------------------------------------")
print("🎥 LIVE FEED is running.")
print("📸 Press 's' to take a High-Quality Snapshot and run Face Recognition.")
print("❌ Press 'ESC' to exit and save attendance to CSV.")
print("---------------------------------------------------")

cv2.namedWindow("Attendance System", cv2.WINDOW_NORMAL)

last_snapshot_time = time.time()

while True:
    ret, frame = cap.read()
    
    if ret and frame is not None:
        cv2.imshow("Attendance System", frame)
    else:
        # If camera isn't ready yet
        time.sleep(0.1)
        continue

    key = cv2.waitKey(1)
    
    current_time = time.time()
    time_since_last_snapshot = current_time - last_snapshot_time
    
    if key == 27: # ESC key
        break
    elif key == ord('s') or time_since_last_snapshot >= 60:
        if time_since_last_snapshot >= 60:
            print("\n⏰ 60 seconds passed! Taking automatic snapshot...")
        else:
            print("\n📸 Manual snapshot requested! Fetching high-quality image...")
            
        last_snapshot_time = current_time # Reset timer
        try:
            req = urllib.request.urlopen(snapshot_url, timeout=5)
            arr = np.array(bytearray(req.read()), dtype=np.uint8)
            hq_img = cv2.imdecode(arr, -1)
            
            if hq_img is not None:
                print("🧠 Processing faces using CNN (this may take a moment)...")
                
                # Run the heavy CNN model on the high-quality snapshot
                results = recognize_faces(hq_img, known_encodings, known_names)
                
                seen_in_snapshot = set()
                
                for (_, _, _, _, name, _) in results:
                    if name != "Unknown" and name not in seen_in_snapshot:
                        attendance_counter[name] += 1
                        seen_in_snapshot.add(name)
                
                print("✅ Snapshot processed! Current Counts:", dict(attendance_counter))
                
                # Draw boxes on the high-quality image and show it in a new window
                hq_display = draw_boxes(hq_img, results)
                cv2.namedWindow("Snapshot Results", cv2.WINDOW_NORMAL)
                cv2.imshow("Snapshot Results", hq_display)
                
            else:
                print("❌ Failed to decode snapshot image.")
        except Exception as e:
            print(f"❌ Error fetching snapshot: {e}")

cap.release()
cv2.destroyAllWindows()

# Final Attendance Logic
unique_names = set(known_names)
final_attendance = {name: "Absent" for name in unique_names}

for name, count in attendance_counter.items():
    if count >= 3: # Marked present if seen at least 3 times
        final_attendance[name] = "Present"

print("\nFinal Attendance:")
for k, v in final_attendance.items():
    print(k, v)

# Save to CSV
df = pd.DataFrame(list(final_attendance.items()), columns=["Name", "Attendance"])
df.to_csv("attendance.csv", index=False)
print("\n✅ Saved to attendance.csv")
