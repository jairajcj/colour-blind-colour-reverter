"""
Quest Vision Web Server
Streams the camera feed with colorblind correction to a local website.
Uses the original ColorBlindnessCorrector logic from colorblind_correction.py.
"""
import cv2
import socket
import threading
import sys
from flask import Flask, Response

# Import the existing corrector class
from colorblind_correction import ColorBlindnessCorrector

app = Flask(__name__)
corrector = ColorBlindnessCorrector()
camera = None

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def generate_frames():
    global camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Apply the correction from your repo
        processed_frame = corrector.process_frame(frame)
        
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    # Return a zero-UI page that just shows the stream
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode/<mode>')
def set_mode(mode):
    if mode in ['normal', 'protanopia', 'deuteranopia', 'tritanopia']:
        corrector.set_mode(mode)
        return f"Mode changed to {mode}"
    return f"Invalid mode: {mode}", 400

def input_listener():
    """Listens for mode changes in the terminal."""
    print("\n[TERMINAL CONTROL ACTIVE]")
    print("Type mode and press Enter: normal, protanopia, deuteranopia, tritanopia\n")
    while True:
        try:
            cmd = sys.stdin.readline().strip().lower()
            if cmd in ['normal', 'protanopia', 'deuteranopia', 'tritanopia']:
                corrector.set_mode(cmd)
            else:
                print(f"Invalid mode: {cmd}. Use: normal, protanopia, deuteranopia, tritanopia")
        except EOFError:
            break

def main():
    global camera
    # Initialize camera with Windows fallback
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    # Start terminal control thread
    threading.Thread(target=input_listener, daemon=True).start()

    ip = get_ip()
    print("\n" + "="*40)
    print(f"WEBSITE URL: http://{ip}:5000")
    print("="*40 + "\n")

    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)

if __name__ == "__main__":
    main()
