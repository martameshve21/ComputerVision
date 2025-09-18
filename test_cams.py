import cv2

def try_open(idx):
    for backend, name in [(cv2.CAP_MSMF, "MSMF"), (cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_ANY, "ANY")]:
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"Camera {idx} opened with {name}")
                cap.release()
                return True
        cap.release()
    print(f"Camera {idx} failed with all backends")
    return False

for i in range(4):  # test first 4 indices
    try_open(i)
