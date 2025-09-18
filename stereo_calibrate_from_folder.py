import cv2, numpy as np, glob, os

# === EDIT THESE ===
pattern_size  = (9, 6)      # inner corners across, down (your checkerboard)
square_size   = 2.5         # size of one square in cm (use the real value!)
left_glob     = "images/stereoLeft/*.*"
right_glob    = "images/stereoright/*.*"
# ===================

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

# 3D points in the checkerboard's coordinate system (Z=0 plane)
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size  # now in cm

left_files  = sorted(glob.glob(left_glob))
right_files = sorted(glob.glob(right_glob))
assert len(left_files) == len(right_files) and len(left_files) > 0, "Left/right count mismatch or empty."

def read_gray(p): 
    img = cv2.imread(p, cv2.IMREAD_COLOR); 
    assert img is not None, f"Cannot read {p}"; 
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

objpoints, imgpoints_l, imgpoints_r = [], [], []
img_size = None
used = 0

for L, R in zip(left_files, right_files):
    gl = read_gray(L); gr = read_gray(R)
    if img_size is None: img_size = gl.shape[::-1]
    ret_l, corners_l = cv2.findChessboardCorners(gl, pattern_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(gr, pattern_size, None)
    if ret_l and ret_r:
        # refine corners
        corners_l = cv2.cornerSubPix(gl, corners_l, (11,11), (-1,-1), criteria)
        corners_r = cv2.cornerSubPix(gr, corners_r, (11,11), (-1,-1), criteria)
        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)
        used += 1

print(f"Found corners in {used}/{len(left_files)} pairs")

# Monocular calibration (gets fx, fy, cx, cy per camera)
ret_l, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints_l, img_size, None, None, criteria=criteria)
ret_r, K2, D2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints_r, img_size, None, None, criteria=criteria)
print("K1:\n", K1)
print("K2:\n", K2)

# Stereo calibration
flags = (cv2.CALIB_FIX_INTRINSIC)  # keep K1,K2 fixed while estimating R,T
retval, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r, K1, D1, K2, D2, img_size,
    criteria=criteria, flags=flags)

print("Stereo RMS:", retval)
print("Baseline (cm) = ", np.linalg.norm(T))  # distance between camera centers

# Rectify
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY)

# Focal lengths to use (pixels) — use P1[0,0] (== P2[0,0] after rectification)
fx_rect = P1[0,0]
fy_rect = P1[1,1]
print("Rectified focal fx (px):", fx_rect, "  fy (px):", fy_rect)
print("cx, cy:", P1[0,2], P1[1,2])

# Save all parameters for your app
np.savez("stereo_params.npz",
         K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, E=E, F=F,
         R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, img_size=img_size, square_size=square_size)

print("Saved -> stereo_params.npz")
