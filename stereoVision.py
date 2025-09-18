import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt

# Function for stereo vision and depth estimation
import triangulation as tri
import calibration

# Mediapipe for face detection
import mediapipe as mp

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# ------------------------------------------------------------------
# Open both cameras (use MSMF backend on Windows for better support)
# You found that indices 0 and 1 work on your system.
# ------------------------------------------------------------------
LEFT_CAM_NAME  = "HD USB Camera"   # or "XCX-1080P"
RIGHT_CAM_NAME = "XCX-1080P"       # pick a second physical USB cam

# --- Cameras: use MSMF for index-based capture on Windows ---
cap_left  = cv2.VideoCapture(0, cv2.CAP_MSMF)   # USB cam
cap_right = cv2.VideoCapture(2, cv2.CAP_MSMF)   # USB cam

if not (cap_left.isOpened() and cap_right.isOpened()):
    raise RuntimeError("Could not open both USB cameras (0 & 2). Try swapping.")

for cap in (cap_left, cap_right):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)


# Stereo vision setup parameters
frame_rate = 30     # Camera frame rate
B = 6            # Distance between the cameras [cm]
f = 4            # Camera lens focal length [mm]
alpha = 56.6        # Camera field of view in the horizontal plane [degrees]

# ------------------------------------------------------------------
# Main program loop with face detector and depth estimation
# ------------------------------------------------------------------
with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:

    while True:
        succes_left, frame_left = cap_left.read()
        succes_right, frame_right = cap_right.read()

        # If cannot catch any frame, continue instead of crashing
        if not succes_left or not succes_right or frame_left is None or frame_right is None:
            print("⚠️ Frame not captured from one of the cameras")
            continue

        # ---------------- CALIBRATION -----------------
        # Uncomment after you have calibration files working
        # frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)
        # ------------------------------------------------

        start = time.time()

        # Convert the BGR image to RGB (for Mediapipe)
        frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

        # Process the image and find faces
        results_right = face_detection.process(frame_right_rgb)
        results_left = face_detection.process(frame_left_rgb)

        # Convert back to BGR for OpenCV drawing
        frame_right = cv2.cvtColor(frame_right_rgb, cv2.COLOR_RGB2BGR)
        frame_left = cv2.cvtColor(frame_left_rgb, cv2.COLOR_RGB2BGR)

        # ---------------- DEPTH CALCULATION -----------------
        center_point_right = None
        center_point_left = None

        if results_right.detections:
            for detection in results_right.detections:
                mp_draw.draw_detection(frame_right, detection)
                bBox = detection.location_data.relative_bounding_box
                h, w, _ = frame_right.shape
                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                center_point_right = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)
                cv2.putText(frame_right, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

        if results_left.detections:
            for detection in results_left.detections:
                mp_draw.draw_detection(frame_left, detection)
                bBox = detection.location_data.relative_bounding_box
                h, w, _ = frame_left.shape
                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                center_point_left = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)
                cv2.putText(frame_left, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

        # If faces detected in both cameras, calculate depth
        if center_point_right and center_point_left:
            depth = tri.find_depth(center_point_right, center_point_left,
                                   frame_right, frame_left, B, f, alpha)
            cv2.putText(frame_right, f"Distance: {round(depth,1)}", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            cv2.putText(frame_left, f"Distance: {round(depth,1)}", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            print("Depth:", round(depth,1))
        else:
            cv2.putText(frame_right, "TRACKING LOST", (75,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            cv2.putText(frame_left, "TRACKING LOST", (75,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

        # ---------------- FPS -----------------
        end = time.time()
        fps = 1 / (end - start)
        cv2.putText(frame_right, f'FPS: {int(fps)}', (20,450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        cv2.putText(frame_left, f'FPS: {int(fps)}', (20,450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

        # Show the frames
        cv2.imshow("frame right", frame_right)
        cv2.imshow("frame left", frame_left)

        # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
