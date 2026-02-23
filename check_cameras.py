import cv2
import sys

def check_cameras(limit=5):
    print("Checking for available cameras...")
    available_cameras = []
    for i in range(limit):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera index {i} is available and working.")
                available_cameras.append(i)
            else:
                print(f"Camera index {i} is opened but failed to read frame.")
            cap.release()
        else:
            print(f"Camera index {i} is not available.")
    return available_cameras

if __name__ == "__main__":
    cams = check_cameras()
    if not cams:
        print("No working cameras found!")
        sys.exit(1)
    else:
        print(f"Found {len(cams)} working camera(s): {cams}")
        sys.exit(0)

