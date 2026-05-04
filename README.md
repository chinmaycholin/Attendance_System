# Smart Attendance System

An automated Face Recognition Attendance System built for classroom use. It uses **OpenCV DNN** for fast face detection and a **ResNet CNN** for high-accuracy identification — all accessible through a clean web dashboard that runs in your browser.

---

## Features

- 🎥 **Live IP Camera Feed** — Zero-lag MJPEG stream displayed directly in the browser
- 🧠 **AI Face Recognition** — OpenCV DNN + ResNet CNN pipeline (no cloud, 100% offline)
- ⏱️ **Auto Snapshots** — Automatically takes a high-quality snapshot every 60 seconds
- 📸 **Manual Snapshot** — On-demand attendance check with a camera flash & toast confirmation
- 📋 **Live Dashboard** — Real-time student attendance list with present/absent status
- 💾 **Timestamped CSV Export** — Each session saved as a uniquely named file (e.g. `attendance_2026-05-04_09-30-00.csv`)
- 🆔 **USN in CSV** — Each student's University Serial Number included in the exported file
- 🗓️ **Date & Time in CSV** — Every CSV records the exact date and time the session was held

---

## CSV Output Format

When you click **■ Stop & Save CSV**, the system generates a file like:

**Filename:** `attendance_2026-05-04_09-30-00.csv`

| USN | Name | Attendance | Date | Time |
|-----|------|-----------|------|------|
| 4MC23CS032 | Chinmay | Present | 04-05-2026 | 09:30 AM |
| 4MC23CS005 | Abhishek | Absent | 04-05-2026 | 09:30 AM |
| 4MC23CS049 | Gagan | Present | 04-05-2026 | 09:30 AM |

> A student is marked **Present** only if they are confidently recognized in **at least 3 separate snapshots** during the session.

---

## Student USN Mapping

The following USN mappings are configured in `app.py` under the `USN_MAP` dictionary:

| Name | USN |
|------|-----|
| Chinmay | 4MC23CS032 |
| Abhishek | 4MC23CS005 |
| Gagan | 4MC23CS049 |

To add more students, open `app.py` and add entries to `USN_MAP`:
```python
USN_MAP = {
    "Chinmay":  "4MC23CS032",
    "Abhishek": "4MC23CS005",
    "Gagan":    "4MC23CS049",
    # "NewStudent": "4MC23CS0XX",
}
```

---

## Installation (First-Time Setup)

### Prerequisites
- Python installed on your computer
- A C++ compiler (e.g. **Visual Studio Build Tools** on Windows) — required by the `face_recognition` library

### Setup Instructions

1. **Open a terminal / command prompt** and navigate to this project folder.
2. **Run the setup script** (only needs to be done once):
   ```bash
   python setup.py
   ```
   *This creates a virtual environment inside the `packages/` folder and installs all required libraries and deep learning models, keeping your global Python clean.*

> ⚠️ **Important:** After setup, always use the virtual environment's Python for all commands:
> ```bash
> .\packages\Scripts\python <script_name>.py
> ```

---

## How to Use

### Step 1: Camera Setup
1. Download the **IP Webcam** app on your Android phone.
2. Open the app, scroll to the bottom, and tap **Start server**.
3. Note the IP address shown on screen (e.g., `http://192.168.1.x:8080`).
4. In the web dashboard, click **⚙️ Settings** and enter your camera URL — no need to edit any code.

### Step 2: Add Training Data
1. Open the `dataset/` folder.
2. Create a sub-folder named after the student (e.g., `dataset/Chinmay/`).
3. Place clear photos of the student inside (include both front-facing and slightly angled shots for best accuracy).

### Step 3: Train the Model
Double-click **`train.bat`** to process the dataset and generate face encodings.

*(Or run manually:)*
```bash
.\packages\Scripts\python train.py
```
> Only needs to be re-run when new students are added to the dataset.

### Step 4: Run the Attendance Dashboard
Double-click **`run.bat`** — your browser will open automatically.

*(Or run manually:)*
```bash
.\packages\Scripts\python app.py
```

#### Dashboard Controls

| Button | Action |
|--------|--------|
| ⚙️ Settings | Set / update the IP Camera URL |
| ▶ Start Attendance | Begin the AI processing loop |
| 📸 Manual Snap | Force an immediate attendance snapshot (shows flash + toast confirmation) |
| ■ Stop & Save CSV | End the session and save the timestamped CSV |
| ⬇ Download CSV | Download the most recently saved CSV file |

---

## Project Structure

```
Attendance chinmay/
├── app.py              # Flask web server & attendance logic
├── main.py             # Standalone OpenCV window version (legacy)
├── train.py            # Face encoding training script
├── recognize.py        # Face detection & recognition pipeline
├── setup.py            # One-time virtual environment + dependency installer
├── run.bat             # Double-click to launch the web dashboard
├── train.bat           # Double-click to re-train the model
├── encodings.pkl       # Generated face encodings (created by train.py)
├── settings.json       # Persisted camera URL setting
├── dataset/            # Student photo folders  (e.g. dataset/Chinmay/)
├── attendance_records/ # All saved CSV files go here (auto-created)
├── packages/           # Virtual environment (created by setup.py)
├── static/             # CSS and frontend assets
└── templates/
    └── index.html      # Web dashboard UI
```
