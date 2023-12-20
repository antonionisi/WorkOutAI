# GymAI

## Description
This script is specifically developed to recognize and count repetitions of predefined exercises using real-time video analysis. It employs the MediaPipe framework for advanced pose detection and the XGBoost machine learning library to classify exercises. The focus is on accurately identifying and counting three types of exercises: lateral raises, squats and curls. It captures video input and processes the images to detect human poses. It then calculates angles at various joints, essential for determining the specific exercise being performed. Utilizing a pre-trained XGBoost model, the script classifies each exercise and keeps track of the repetitions.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)
- XGBoost (`xgboost`)

Ensure you have the above libraries installed in your Python environment.

## Setup
- Place the `model.json` file, which contains the pre-trained XGBoost model, in the same directory as the script.
- In line 65, “cv2.VideoCapture(1)”, the number within the brackets specifies the camera device to be used. This should generally be set to 0 for Windows and Linux systems and to 1 for Mac users. This is to ensure that the script captures video from the correct camera device connected to your computer.
- Adapt lines 122-123 in the script to optimize text visualization on the video feed. These lines adjust the size and position of the text based on your webcam's resolution. It ensures that the exercise counts are displayed clearly and are scaled appropriately according to different screen sizes.

## How to Run
1. Execute the script in a Python environment.
2. A video feed will open using your default camera.
3. Perform exercises in view of the camera. The script will analyze your pose and count the repetitions of different exercises.

## Functions Description
- `calculate_angle(a, b, c)`: Calculates the angle at joint 'b', given coordinates of points 'a', 'b' and 'c'.
- `extract_angles(landmarks)`: Extracts and calculates relevant angles from the pose landmarks for pose analysis.
- The main loop captures video frames, processes them for pose detection, calculates angles and uses the XGBoost model to classify and count exercises.

## Note
- The model's accuracy and script performance depend on the lighting, camera quality and how distinctly the exercises are performed.
- Ensure sufficient space and safety while performing exercises in front of the camera.