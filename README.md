# StealthModeAssignment
Real-time and video-based player tracking using YOLOv8 and centroid logic. Tracks players by position, jersey color, and height. Outputs annotated video and CSV logs. Handles ID assignment, avoids overlaps, and supports future upgrades with DeepSORT or advanced tracking methods.


Setup Instructions

Google Colab Users
(Benefit: Free limited GPU inside Google Colab)

    Save the required files to your Google Drive:

        requirements.txt

        15sec_input_720p.mp4

        best.pt (YOLOv8 model)

    Open Google Colab with the same Gmail ID as your Google Drive.

    Mount Google Drive in your Colab notebook to access the project files.

    Install the requirements from requirements.txt.

    Specify the output path where annotated video and CSV logs will be saved.

 Jupyter Notebook Users

(Benefit: Can utilize system GPU for faster performance)

    No need to use Google Drive — access files directly from your system using absolute or relative paths.

    Required files (15sec_input_720p.mp4, requirements.txt) can be downloaded from this repository.

    If the model file best.pt is available within YOLO's model directory, you can reference it directly by name (e.g., best.pt) without specifying the full path.

 Real-Time Execution via Code Editor

    Use the real-time script included in the repository.

    You can reference system paths directly for both input and output.

    It is recommended to create a virtual environment to isolate this project and its dependencies from others.

Virtual Environment Setup
For Windows:

# Create virtual environment
python -m venv env_name

# Activate environment
env_name\Scripts\activate

# Deactivate
deactivate

For Linux/Ubuntu:

# Create virtual environment
python3 -m venv env_name

# Activate environment
source env_name/bin/activate

# Deactivate
deactivate

 Support

If you need further assistance, feel free to contact me at:
📧 2023nitsgr245@nitsri.ac.in



