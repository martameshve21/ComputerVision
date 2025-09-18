Stereo Vision Depth Estimation with Face Detection

This project implements a stereo vision system using two USB cameras to estimate the depth of objects (or faces) in real-time. It combines OpenCV for stereo camera calibration and image processing with MediaPipe for face detection.

Features

Capture video from two calibrated USB cameras.

Detect faces in both camera streams using Google MediaPipe.

Estimate the 3D depth of detected faces using stereo vision geometry.

Includes camera calibration pipeline with chessboard images to compute:

Intrinsic camera parameters (focal length, principal point).

Distortion coefficients.

Stereo extrinsics (rotation and translation between cameras).

Real-time visualization with FPS counter and distance overlay
