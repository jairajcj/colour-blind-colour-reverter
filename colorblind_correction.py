"""
Color Blindness Correction System
Real-time camera application to help colorblind individuals see colors more accurately.
Supports Protanopia, Deuteranopia, and Tritanopia correction.
"""

import cv2
import numpy as np
import argparse
import time
from typing import Tuple, Union


class ColorBlindnessCorrector:
    """
    Implements Daltonization algorithms for color blindness correction.
    Uses LMS (Long, Medium, Short wavelength) color space transformation.
    """
    
    def __init__(self):
        # RGB to LMS transformation matrix
        self.rgb2lms = np.array([
            [17.8824, 43.5161, 4.11935],
            [3.45565, 27.1554, 3.86714],
            [0.0299566, 0.184309, 1.46709]
        ])
        
        # LMS to RGB transformation matrix (inverse)
        self.lms2rgb = np.linalg.inv(self.rgb2lms)
        
        # Simulation matrices for different types of color blindness
        # Protanopia (red-blind) - missing L-cones
        self.protanopia_sim = np.array([
            [0.0, 2.02344, -2.52581],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Deuteranopia (green-blind) - missing M-cones
        self.deuteranopia_sim = np.array([
            [1.0, 0.0, 0.0],
            [0.494207, 0.0, 1.24827],
            [0.0, 0.0, 1.0]
        ])
        
        # Tritanopia (blue-blind) - missing S-cones
        self.tritanopia_sim = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-0.395913, 0.801109, 0.0]
        ])
        
        # Precompute combined matrices for optimization
        self.combined_matrices = self._precompute_matrices()
        self.current_mode = 'normal'
        
    def _precompute_matrices(self):
        """Precompute the combined RGB-to-RGB correction matrices."""
        matrices = {}
        identity = np.eye(3)
        
        for cb_type in ['protanopia', 'deuteranopia', 'tritanopia']:
            # Simulation matrix
            if cb_type == 'protanopia':
                sim = self.protanopia_sim
                err_corr = np.array([
                    [0, 0, 0],
                    [0.7, 1, 0],
                    [0.7, 0, 1]
                ])
            elif cb_type == 'deuteranopia':
                sim = self.deuteranopia_sim
                err_corr = np.array([
                    [1, 0.7, 0],
                    [0, 0, 0],
                    [0, 0.7, 1]
                ])
            else: # tritanopia
                sim = self.tritanopia_sim
                err_corr = np.array([
                    [1, 0, 0.7],
                    [0, 1, 0.7],
                    [0, 0, 0]
                ])
            
            # Combined matrix = M_rgb2lms * (I + (I - M_sim) * M_err_corr) * M_lms2rgb
            # Note: The math above assumes row vectors (RGB * M). 
            # If using cv2.transform(src, M), src is (H,W,3) and M is (3,3) or (3,4).
            # cv2.transform(src, M) computes dst(x,y,0) = M[0,0]*src(x,y,0) + M[0,1]*src(x,y,1) + M[0,2]*src(x,y,2)
            # This is equivalent to src(x,y) * M.T
            
            # (I - M_sim)
            diff = identity - sim
            # (I - M_sim) * M_err_corr
            correction = diff @ err_corr
            # (I + correction)
            total_lms_transform = identity + correction
            
            # Final matrix: M_rgb2lms @ total_lms_transform @ M_lms2rgb
            combined = self.rgb2lms @ total_lms_transform @ self.lms2rgb
            
            # Store transposed for cv2.transform or non-transposed for @
            matrices[cb_type] = combined
            
        return matrices
        
    def rgb_to_lms(self, rgb_image: np.ndarray) -> np.ndarray:
        """Convert RGB image to LMS color space."""
        # Normalize to 0-1 range
        rgb_normalized = rgb_image.astype(np.float32) / 255.0
        
        # Reshape for matrix multiplication
        h, w, c = rgb_normalized.shape
        rgb_reshaped = rgb_normalized.reshape(-1, 3)
        
        # Apply transformation
        lms = rgb_reshaped @ self.rgb2lms.T
        
        return lms.reshape(h, w, 3)
    
    def lms_to_rgb(self, lms_image: np.ndarray) -> np.ndarray:
        """Convert LMS image back to RGB color space."""
        h, w, c = lms_image.shape
        lms_reshaped = lms_image.reshape(-1, 3)
        
        # Apply transformation
        rgb = lms_reshaped @ self.lms2rgb.T
        
        # Clip and denormalize
        rgb = np.clip(rgb, 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
        
        return rgb.reshape(h, w, 3)
    
    def simulate_colorblindness(self, lms_image: np.ndarray, cb_type: str) -> np.ndarray:
        """Simulate how a colorblind person would see the image."""
        h, w, c = lms_image.shape
        lms_reshaped = lms_image.reshape(-1, 3)
        
        if cb_type == 'protanopia':
            sim_matrix = self.protanopia_sim
        elif cb_type == 'deuteranopia':
            sim_matrix = self.deuteranopia_sim
        elif cb_type == 'tritanopia':
            sim_matrix = self.tritanopia_sim
        else:
            return lms_image
        
        # Apply simulation
        lms_sim = lms_reshaped @ sim_matrix.T
        
        return lms_sim.reshape(h, w, 3)
    
    def daltonize(self, rgb_image: np.ndarray, cb_type: str) -> np.ndarray:
        """
        Apply optimized Daltonization algorithm to correct for color blindness.
        Uses a precomputed matrix for single-pass transformation.
        """
        if cb_type not in self.combined_matrices:
            return rgb_image
            
        # Use cv2.transform for high performance
        # Input: 0-255 uint8, Output: float32, then clip and convert back
        mat = self.combined_matrices[cb_type]
        
        # cv2.transform expects a 3x3 matrix for (N,3) data
        # It performs: dst[i] = mat[i,0]*src[0] + mat[i,1]*src[1] + mat[i,2]*src[2]
        # This is exactly what we want if mat is our combined matrix
        
        # We need to normalize/denormalize because the matrix was derived for 0-1 range
        # However, since it's a linear transformation, we can just apply it and clip
        # dst = src * M is same as (src/255 * M) * 255
        
        corrected = cv2.transform(rgb_image.astype(np.float32), mat)
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        return corrected
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame based on current mode."""
        if self.current_mode == 'normal':
            return frame
        elif self.current_mode in ['protanopia', 'deuteranopia', 'tritanopia']:
            return self.daltonize(frame, self.current_mode)
        else:
            return frame
    
    def set_mode(self, mode: str):
        """Set the correction mode."""
        valid_modes = ['normal', 'protanopia', 'deuteranopia', 'tritanopia']
        if mode in valid_modes:
            self.current_mode = mode
            print(f"Mode changed to: {mode.upper()}")
        else:
            print(f"Invalid mode: {mode}")


class ColorBlindnessApp:
    """Main application for real-time color blindness correction."""
    
    def __init__(self, camera_source: Union[int, str] = 0, width: int = 640, height: int = 480):
        """
        Initialize the application.
        
        Args:
            camera_source: Can be:
                - int: Local webcam index (0, 1, 2...)
                - str: IP camera URL, e.g.:
                    - 'http://192.168.1.100:8080/video'  (IP Webcam Android)
                    - 'rtsp://user:pass@192.168.1.100:554/stream1'  (RTSP)
                    - 'http://192.168.1.100/mjpg/video.mjpg'  (MJPEG)
            width: Frame width (applied to local cameras only)
            height: Frame height (applied to local cameras only)
        """
        self.corrector = ColorBlindnessCorrector()
        self.camera_source = camera_source
        self.is_ip_camera = isinstance(camera_source, str)
        self.width = width
        self.height = height
        self.cap = None
        self.fps = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
    def initialize_camera(self) -> bool:
        """Initialize camera capture (local or IP)."""
        source_label = self.camera_source if self.is_ip_camera else f"Local camera #{self.camera_source}"
        print(f"Connecting to: {source_label}")
        
        self.cap = cv2.VideoCapture(self.camera_source)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera source: {source_label}")
            if self.is_ip_camera:
                print("  -> Check that the IP address is correct and the camera is on the same network.")
                print("  -> Common URL formats:")
                print("       Android IP Webcam: http://<IP>:8080/video")
                print("       RTSP:              rtsp://user:pass@<IP>:554/stream1")
                print("       MJPEG:             http://<IP>/mjpg/video.mjpg")
            return False
        
        # Set camera properties (mainly effective for local cameras)
        if not self.is_ip_camera:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # For IP cameras, set buffer size to 1 to reduce latency
        if self.is_ip_camera:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.reconnect_attempts = 0
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized successfully ({actual_w}x{actual_h})")
        return True
    
    def reconnect_camera(self) -> bool:
        """Attempt to reconnect to an IP camera after a dropped connection."""
        self.reconnect_attempts += 1
        if self.reconnect_attempts > self.max_reconnect_attempts:
            print(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached. Exiting.")
            return False
        
        print(f"Connection lost. Reconnecting... (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
        if self.cap:
            self.cap.release()
        time.sleep(2)  # Wait before retry
        return self.initialize_camera()
    
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI elements on the frame."""
        # Create a copy to draw on
        display_frame = frame.copy()
        
        # Draw semi-transparent background for text
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        mode_text = f"Mode: {self.corrector.current_mode.upper()}"
        fps_text = f"FPS: {self.fps:.1f}"
        
        cv2.putText(display_frame, mode_text, (20, 40), font, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, fps_text, (20, 70), font, 0.6, (255, 255, 255), 1)
        
        # Camera source label
        src_label = f"Source: {'IP Camera' if self.is_ip_camera else 'Local Webcam'}"        
        cv2.putText(display_frame, src_label, (20, 100), font, 0.5, (200, 200, 200), 1)
        
        # Draw controls panel
        cv2.rectangle(overlay, (10, 120), (500, 260), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        controls = [
            "N - Normal (no correction)",
            "P - Protanopia (red-blind)",
            "D - Deuteranopia (green-blind)",
            "T - Tritanopia (blue-blind)",
            "Q - Quit"
        ]
        
        y_offset = 145
        for control in controls:
            cv2.putText(display_frame, control, (20, y_offset), font, 0.45, (200, 200, 200), 1)
            y_offset += 25
        
        return display_frame
    
    def run(self):
        """Main application loop."""
        if not self.initialize_camera():
            return
        
        print("\n" + "="*50)
        print("Color Blindness Correction System")
        print("="*50)
        print("\nControls:")
        print("  N - Normal mode (no correction)")
        print("  P - Protanopia correction (red-blind)")
        print("  D - Deuteranopia correction (green-blind)")
        print("  T - Tritanopia correction (blue-blind)")
        print("  Q - Quit")
        print("\nStarting camera feed...\n")
        
        # For FPS calculation
        prev_time = time.time()
        
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            
            if not ret:
                if self.is_ip_camera:
                    # Try to reconnect for IP cameras
                    if not self.reconnect_camera():
                        break
                    continue
                else:
                    print("Error: Failed to capture frame")
                    break
            
            # Process frame
            processed_frame = self.corrector.process_frame(frame)
            
            # Calculate FPS
            current_time = time.time()
            self.fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # Draw UI
            display_frame = self.draw_ui(processed_frame)
            
            # Display
            cv2.imshow('Color Blindness Correction', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\nExiting...")
                break
            elif key == ord('n') or key == ord('N'):
                self.corrector.set_mode('normal')
            elif key == ord('p') or key == ord('P'):
                self.corrector.set_mode('protanopia')
            elif key == ord('d') or key == ord('D'):
                self.corrector.set_mode('deuteranopia')
            elif key == ord('t') or key == ord('T'):
                self.corrector.set_mode('tritanopia')
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed successfully")


def main():
    """Entry point for the application."""
    parser = argparse.ArgumentParser(
        description='Color Blindness Correction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python colorblind_correction.py                          # Use default webcam
  python colorblind_correction.py --camera 1               # Use second webcam
  python colorblind_correction.py --ip http://192.168.1.100:8080/video   # Android IP Webcam
  python colorblind_correction.py --ip rtsp://user:pass@192.168.1.5:554/stream1  # RTSP camera
        """
    )
    parser.add_argument('--camera', type=int, default=0,
                        help='Local camera index (default: 0)')
    parser.add_argument('--ip', type=str, default=None,
                        help='IP camera URL (overrides --camera if set)')
    parser.add_argument('--width', type=int, default=640,
                        help='Frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Frame height (default: 480)')
    
    args = parser.parse_args()
    
    # Determine camera source
    if args.ip:
        camera_source = args.ip
        print(f"Using IP camera: {args.ip}")
    else:
        camera_source = args.camera
        print(f"Using local camera: {args.camera}")
    
    app = ColorBlindnessApp(camera_source=camera_source, width=args.width, height=args.height)
    app.run()


if __name__ == "__main__":
    main()
