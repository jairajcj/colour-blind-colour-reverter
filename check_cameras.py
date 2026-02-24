import cv2
import sys

def check_cameras(limit=10):
    print("Checking for available cameras using multiple backends...")
    available_cameras = []
    
    backends = [
        ("Default", None),
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF)
    ]
    
    for name, backend in backends:
        print(f"\nTesting backend: {name}")
        for i in range(limit):
            if backend is not None:
                cap = cv2.VideoCapture(i, backend)
            else:
                cap = cv2.VideoCapture(i)
                
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"  [SUCCESS] Camera {i} available.")
                    if (i, name) not in available_cameras:
                        available_cameras.append((i, name))
                else:
                    print(f"  [WARN] Camera {i} opened but failed to read.")
                cap.release()
    return available_cameras

if __name__ == "__main__":
    cams = check_cameras()
    if not cams:
        print("\nNo working cameras found!")
    else:
        print(f"\nFound working camera(s):")
        for idx, backend in cams:
            print(f" - Index {idx} with {backend}")
