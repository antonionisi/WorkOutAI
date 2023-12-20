##############################################################################################################
#
# Work-OutAI Computer-Vision Project
#
# This script is the main script of the project.
# Authors: Andrea Zoccante, Antonio Nisi
# Date: 15-12-2023
#
##############################################################################################################


#Import libraries
import cv2
import mediapipe as mp
import numpy as np

#Inizialize pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#Import XGBoost model
import xgboost as xgb
model = xgb.Booster()
model.load_model('model.json')

def calculate_angle(a,b,c):
    #Extract vertices
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    #Compute radiants between the edges
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    
    #Convert in degrees
    angle = np.abs(radians*180.0/np.pi)
    
    #Adjust the sign
    if angle >180.0:
        angle = 360-angle
        
    return angle

def extract_angles(landmarks):
    # Get coordinates
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    
    # Calculate angles
    angle_left_elbow = calculate_angle(left_shoulder, left_elbow, left_wrist)
    angle_right_elbow = calculate_angle(right_shoulder, right_elbow, right_wrist)
    angle_left_shoulder = calculate_angle(left_elbow, left_shoulder, left_hip)
    angle_right_shoulder = calculate_angle(right_elbow, right_shoulder, right_hip)
    angle_left_hip = calculate_angle(left_shoulder, left_hip, left_knee)
    angle_right_hip = calculate_angle(right_shoulder, right_hip, right_knee)
    angle_left_knee = calculate_angle(left_ankle, left_knee, left_hip)
    angle_right_knee = calculate_angle(right_ankle, right_knee, right_hip)
    
    #Collect angles in a vector
    angles = [angle_left_elbow, angle_right_elbow, angle_left_shoulder, angle_right_shoulder, angle_left_hip, angle_right_hip, angle_left_knee, angle_right_knee]
    
    return angles


# Video feed
cap = cv2.VideoCapture(1)


# Counter variables
counter_lr = 0
counter_squat = 0
counter_curl = 0
stage = None

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection 
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            #Extract landmarks and compute angles
            landmarks = results.pose_landmarks.landmark
            angles = extract_angles(landmarks)
            
            #Evaluate the state of the pose
            comf = (angles[0] > 150) & (angles[1] > 150) & (angles[2] < 30) & (angles[3] < 30) & (angles[4] > 150) & (angles[5] > 150) & (angles[6] > 150) & (angles[7] > 150)
            notcomf = (angles[0] < 30) | (angles[1] < 30) | (angles[2] > 80) | (angles[3] > 80) | (angles[4] < 50) | (angles[5] < 50) | (angles[6] < 110) | (angles[7] < 110)  
            
            # Exercises counter logic
            if comf:
                stage = "down"
            if notcomf:
                if stage =='down':
                    # Model evaluation of the exercise
                    data = xgb.DMatrix([angles])
                    exercise = np.argmax(model.predict(data))

                    # Update counters
                    if exercise == 0:
                        counter_lr += 1
                    elif exercise == 1:
                        counter_squat +=1
                    else:
                        counter_curl += 1
                    stage="up"
                       
        except:
            pass
        
        
        # Get the resolution of the webcam
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Scale factors
        text_scale = width / 640
        position_scale = height / 480
        
        # Rep data
        cv2.putText(image, "lateral rises: "+str(counter_lr),
                    (int(10 * position_scale), int(40 * position_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255,255,255), int(2 * text_scale), cv2.LINE_AA)
        cv2.putText(image, "squats: "+str(counter_squat),
                    (int(10 * position_scale), int(80 * position_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255,255,255), int(2 * text_scale), cv2.LINE_AA)
        cv2.putText(image, "curls: "+str(counter_curl),
                    (int(10 * position_scale), int(120 * position_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255,255,255), int(2 * text_scale), cv2.LINE_AA)

        
        # Show the image
        cv2.imshow('Work-OutAI', image)

        # Break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()