# Smart Attendance System

This is an automated Face Recognition Attendance System that uses OpenCV's Deep Neural Networks (DNN) for blazing-fast face detection and a heavy ResNet CNN for high-accuracy identification.

## How to Install (For First-Time Setup)

If you have just downloaded this project to a new computer, follow these simple steps to install everything you need.

### Prerequisites
Make sure you have Python installed on your computer. You will also need a C++ compiler installed (like Visual Studio Build Tools on Windows) for the `face_recognition` library to compile properly.

### Setup Instructions

1. **Open your terminal / command prompt** and navigate to this folder.
2. **Run the Setup Script**:
   ```bash
   python setup.py
   ```
   *This script is smart! It will automatically build a secure virtual environment and install all required Python libraries and deep learning models precisely into the `packages` folder, keeping your global Python clean.*

---

## How to Use

### Step 1: Camera Setup
1. Download the **IP Webcam** app on your phone (available on Android).
2. Open the app, scroll to the very bottom, and tap **Start server**.
3. The app will display an IP address on your phone screen (e.g., `http://192.168.1.x:8080`).
4. Open the `main.py` file in this project and look for the `base_url` variable near the top (around line 50).
5. Change that URL to match the exact IP address shown on your phone screen.

### Step 2: Add Training Data
1. Open the `dataset/` folder.
2. Create a new folder with the name of the student (e.g., `dataset/Chinmay/`).
3. Place a few high-quality pictures of the student inside that folder. Make sure to include some straight-on photos and some side-profile photos for maximum accuracy!

### Step 3: Train the System
Run the training script to process the new dataset and generate the mathematical encodings. Simply **double-click the `train.bat` file**!

*(Alternatively, you can run the following command in your terminal):*
```bash
.\packages\Scripts\python train.py
```
*Note: You only need to run this when you add new people to the dataset.*

### Step 4: Run the Attendance Dashboard
Simply **double-click the `run.bat` file** in your folder!
*(Alternatively, you can run `.\packages\Scripts\python app.py` in your terminal).*

- Your web browser will automatically open straight to the beautiful attendance dashboard!
- Click "**⚙️ Settings**" to update your IP Camera URL if needed.
- Click "**▶ Start Attendance**" to begin the background AI processing.
- The system will automatically process the camera feed every 60 seconds and update the live dashboard.
- You can also click "**📸 Manual Snap**" at any time to force an immediate, on-demand attendance check.
- When class is over, click "**■ Stop & Save CSV**", then click "**⬇ Download CSV**" to get your file.

*(Note: A student is only marked "Present" if they are confidently recognized in at least 3 separate snapshots during the session).*
