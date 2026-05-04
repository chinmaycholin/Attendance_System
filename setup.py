import os
import subprocess
import sys
import urllib.request
import venv

def install_packages():
    venv_dir = "packages"
    
    # Auto-create the virtual environment if it doesn't exist yet
    if not os.path.exists(os.path.join(venv_dir, "Scripts")) and not os.path.exists(os.path.join(venv_dir, "bin")):
        print(f"🛠️ Creating virtual environment inside '{venv_dir}' folder...")
        venv.create(venv_dir, with_pip=True)
        
    # Get the exact path to the virtual environment's pip
    if os.name == 'nt': # Windows
        pip_exe = os.path.join(venv_dir, "Scripts", "pip.exe")
    else: # Mac/Linux
        pip_exe = os.path.join(venv_dir, "bin", "pip")

    print(f"📦 Installing required packages securely into '{venv_dir}' folder...")
    packages = ["face_recognition", "opencv-python", "numpy", "pandas", "Pillow", "Flask"]
    
    try:
        # Install cmake first as it's required to build dlib/face_recognition
        print("⚙️ Installing CMake...")
        subprocess.check_call([pip_exe, "install", "cmake"])
        
        print("📥 Installing main dependencies...")
        subprocess.check_call([pip_exe, "install"] + packages)
        print("✅ Packages installed successfully!\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages. Error: {e}")
        sys.exit(1)

def download_file(url, save_path):
    print(f"⬇️ Downloading {os.path.basename(save_path)}...")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"✅ Successfully downloaded {os.path.basename(save_path)}")
    except Exception as e:
        print(f"❌ Failed to download {os.path.basename(save_path)}: {e}")

def download_models():
    print("🤖 Downloading OpenCV Face Detection Models...")
    models_dir = "packages"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    prototxt_path = os.path.join(models_dir, "deploy.prototxt")
    caffemodel_path = os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(prototxt_path):
        download_file(prototxt_url, prototxt_path)
    else:
        print("✅ deploy.prototxt already exists.")

    if not os.path.exists(caffemodel_path):
        download_file(caffemodel_url, caffemodel_path)
    else:
        print("✅ res10_300x300_ssd_iter_140000.caffemodel already exists.")
    print("\n✅ All models are ready!")

if __name__ == "__main__":
    print("🚀 Starting Full System Setup...\n")
    
    install_packages()
    download_models()
    
    print("\n🎉 Setup Complete! You can now run the attendance system.")
