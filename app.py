import os
import cv2
import pickle
import time
import json
from datetime import datetime
import threading
import urllib.request
import numpy as np
import pandas as pd
import webbrowser
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, Response, send_file
from recognize import recognize_faces, draw_boxes

app = Flask(__name__)

# Load encodings
try:
    with open("encodings.pkl", "rb") as f:
        data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]
    unique_names = set(known_names)
except FileNotFoundError:
    print("Error: encodings.pkl not found. Please run train.py first.")
    known_encodings, known_names, unique_names = [], [], set()

# USN mapping  (add more students here as needed)
USN_MAP = {
    "Chinmay":  "4MC23CS032",
    "Abhishek": "4MC23CS005",
    "Gagan":    "4MC23CS049",
}

# Folder where all CSV files are stored
CSV_DIR = "attendance_records"
os.makedirs(CSV_DIR, exist_ok=True)

# Global state
attendance_counter = defaultdict(int)
final_attendance = {name: "Absent" for name in unique_names}
is_running = False
camera_url = "http://192.168.1.3:8080"
attendance_thread = None
last_snapshot_time = 0
session_start_time = None   # Set when attendance starts
last_csv_filename = os.path.join(CSV_DIR, "attendance.csv")  # Tracks the most recently saved file

# Load settings
if os.path.exists("settings.json"):
    with open("settings.json", "r") as f:
        settings = json.load(f)
        camera_url = settings.get("camera_url", camera_url)

def run_attendance_loop():
    global is_running, attendance_counter, final_attendance, last_snapshot_time
    
    last_snapshot_time = time.time() - 60 # Force immediate run on start
    
    while is_running:
        current_time = time.time()
        
        if current_time - last_snapshot_time >= 60:
            snapshot_url = f"{camera_url}/shot.jpg"
            print("\n📸 Fetching automatic snapshot for attendance...")
            try:
                req = urllib.request.urlopen(snapshot_url, timeout=5)
                arr = np.array(bytearray(req.read()), dtype=np.uint8)
                hq_img = cv2.imdecode(arr, -1)
                
                if hq_img is not None:
                    print("🧠 Processing faces via OpenCV DNN -> ResNet CNN...")
                    results = recognize_faces(hq_img, known_encodings, known_names)
                    seen_in_snapshot = set()
                    
                    for (_, _, _, _, name, _) in results:
                        if name != "Unknown" and name not in seen_in_snapshot:
                            attendance_counter[name] += 1
                            seen_in_snapshot.add(name)
                            
                    # Update final attendance instantly
                    for name, count in attendance_counter.items():
                        if count >= 3:
                            final_attendance[name] = "Present"
                    print(f"✅ Current Counts: {dict(attendance_counter)}")
                            
            except Exception as e:
                print(f"❌ Error fetching snapshot: {e}")
                
            last_snapshot_time = current_time
            
        time.sleep(1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    return jsonify({
        "is_running": is_running,
        "camera_url": camera_url,
        "attendance": final_attendance,
        "counts": dict(attendance_counter)
    })

@app.route("/api/start", methods=["POST"])
def start_attendance():
    global is_running, attendance_thread, attendance_counter, final_attendance, session_start_time
    if not is_running:
        # Reset counters and record session start time
        attendance_counter = defaultdict(int)
        final_attendance = {name: "Absent" for name in unique_names}
        session_start_time = datetime.now()
        
        is_running = True
        attendance_thread = threading.Thread(target=run_attendance_loop)
        attendance_thread.daemon = True
        attendance_thread.start()
    return jsonify({"success": True})

@app.route("/api/stop", methods=["POST"])
def stop_attendance():
    global is_running, last_csv_filename
    is_running = False
    
    # Build timestamped filename inside attendance_records/
    # e.g. attendance_records/attendance_2026-05-04_21-30-00.csv
    now = session_start_time or datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    csv_name = f"attendance_{date_str}_{time_str}.csv"
    last_csv_filename = os.path.join(CSV_DIR, csv_name)
    
    # Add USN, Date and Time columns so the teacher has full info
    rows = []
    for name, status in final_attendance.items():
        rows.append({
            "USN":        USN_MAP.get(name, "N/A"),
            "Name":       name,
            "Attendance": status,
            "Date":       now.strftime("%d-%m-%Y"),
            "Time":       now.strftime("%I:%M %p")   # e.g. 09:30 AM
        })
    
    df = pd.DataFrame(rows, columns=["USN", "Name", "Attendance", "Date", "Time"])
    df.to_csv(last_csv_filename, index=False)
    print(f"\n✅ Saved {last_csv_filename}")
    
    return jsonify({"success": True})

@app.route("/api/manual_snap", methods=["POST"])
def manual_snap():
    global last_snapshot_time
    if is_running:
        # Forcing the timer to 0 immediately triggers the loop thread to take a snapshot
        last_snapshot_time = 0 
        return jsonify({"success": True})
    return jsonify({"success": False}), 400

@app.route("/api/settings", methods=["POST"])
def save_settings():
    global camera_url
    data = request.json
    new_url = data.get("camera_url")
    if new_url:
        camera_url = new_url
        with open("settings.json", "w") as f:
            json.dump({"camera_url": camera_url}, f)
        return jsonify({"success": True})
    return jsonify({"success": False}), 400

@app.route('/download_csv')
def download_csv():
    if os.path.exists(last_csv_filename):
        return send_file(last_csv_filename, as_attachment=True, download_name=last_csv_filename)
    return "CSV not found. Please stop the attendance session first to generate the file.", 404

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    # Wait 1.5 seconds for Flask to fully start, then open the browser automatically
    threading.Timer(1.5, open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
