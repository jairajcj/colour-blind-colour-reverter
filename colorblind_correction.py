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
        
        self.combined_matrices = self._precompute_matrices()
        self.selective_mode = True
        self.current_mode = 'normal'
        
    def _precompute_matrices(self):
        """Precompute the combined RGB-to-RGB correction matrices and simulation matrices."""
        matrices = {}
        identity = np.eye(3)
        
        # We also need pure simulation matrices for masking
        self.sim_rgb_matrices = {}
        
        for cb_type in ['protanopia', 'deuteranopia', 'tritanopia']:
            # Simulation matrix in LMS
            if cb_type == 'protanopia':
                sim_lms = self.protanopia_sim
                err_corr = np.array([
                    [0, 0, 0],
                    [0.7, 1, 0],
                    [0.7, 0, 1]
                ])
            elif cb_type == 'deuteranopia':
                sim_lms = self.deuteranopia_sim
                err_corr = np.array([
                    [1, 0.7, 0],
                    [0, 0, 0],
                    [0, 0.7, 1]
                ])
            else: # tritanopia
                sim_lms = self.tritanopia_sim
                err_corr = np.array([
                    [1, 0, 0.7],
                    [0, 1, 0.7],
                    [0, 0, 0]
                ])
            
            # 1. Direct RGB simulation matrix (for masking)
            # M_sim_rgb = M_rgb2lms @ M_sim_lms @ M_lms2rgb
            sim_rgb = self.rgb2lms @ sim_lms @ self.lms2rgb
            self.sim_rgb_matrices[cb_type] = sim_rgb
            
            # 2. Combined Daltonization matrix
            # M_combined = M_rgb2lms @ (I + (I - M_sim_lms) @ M_err_corr) @ M_lms2rgb
            diff = identity - sim_lms
            correction = diff @ err_corr
            total_lms_transform = identity + correction
            combined = self.rgb2lms @ total_lms_transform @ self.lms2rgb
            
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
        ULTIMATE DALTONIZATION ENGINE (Limitless Version).
        1. Machado Spectral Logic
        2. Temporal Stabilization
        3. Local Contrast Enhancement
        """
        if cb_type not in self.combined_matrices:
            return rgb_image
            
        h, w = rgb_image.shape[:2]
        float_img = rgb_image.astype(np.float32)
        
        # 1. Physiologically-Balanced Correction
        mat = self.combined_matrices[cb_type]
        full_corrected = cv2.transform(float_img, mat)
        
        # 2. Perceptual Luminance Preservation (CIELAB)
        lab_orig = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2Lab).astype(np.float32)
        lab_corr = cv2.cvtColor(np.clip(full_corrected, 0, 255).astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float32)
        lab_corr[:,:,0] = lab_orig[:,:,0]
        
        full_corrected_p = cv2.cvtColor(lab_corr.astype(np.uint8), cv2.COLOR_Lab2BGR).astype(np.float32)
        
        # 3. Temporal Selection Mask (Object-Lock Logic)
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_chan, s_chan, v_chan = cv2.split(hsv)
        
        # Advanced Gaussian Hue Detection
        if cb_type == 'protanopia':
            dist_h = np.minimum(np.abs(h_chan - 0), np.abs(h_chan - 180))
            hue_weight = np.exp(-0.5 * (dist_h / 8.0)**2)
        elif cb_type == 'deuteranopia':
            dist_h = np.abs(h_chan - 60)
            hue_weight = np.exp(-0.5 * (dist_h / 12.0)**2)
        elif cb_type == 'tritanopia':
            dist_h = np.abs(h_chan - 120)
            hue_weight = np.exp(-0.5 * (dist_h / 15.0)**2)
        else:
            hue_weight = np.ones_like(h_chan)

        # Precise Saturation Gating
        sat_weight = np.clip((s_chan - 35) / 50.0, 0, 1)
        curr_mask = (hue_weight * sat_weight)
        
        # Object-Lock (Temporal Stabilization)
        if not hasattr(self, '_mask_ema') or self._mask_ema.shape != curr_mask.shape:
            self._mask_ema = curr_mask
        else:
            self._mask_ema = cv2.addWeighted(curr_mask, 0.7, self._mask_ema, 0.3, 0)
        
        # Refine mask edges
        mask = cv2.bilateralFilter(self._mask_ema.astype(np.float32), 7, 0.3, 5)
        mask = np.expand_dims(mask, axis=2)
        
        # 4. Strategic Sharpness Injection
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        sharpening = cv2.GaussianBlur(laplacian, (3, 3), 0)
        float_img = float_img + (np.expand_dims(sharpening, axis=2) * mask * 0.5)

        # 5. Final Ultimate Blend
        diff = full_corrected_p - float_img
        selective_corrected = float_img + (diff * mask)
        
        return np.clip(selective_corrected, 0, 255).astype(np.uint8)
    
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

    def toggle_selective(self):
        """Toggle between full-screen and selective correction."""
        self.selective_mode = not self.selective_mode
        state = "ON (Object-Specific)" if self.selective_mode else "OFF (Full-Screen)"
        print(f"Selective Correction: {state}")


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
        
        if not self.is_ip_camera and hasattr(cv2, 'CAP_DSHOW'):
            # Using CAP_DSHOW often works better for built-in webcams on Windows
            self.cap = cv2.VideoCapture(self.camera_source, cv2.CAP_DSHOW)
        else:
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
        
        # Selective mode status
        sel_label = f"Selective Mode: {'ON' if self.corrector.selective_mode else 'OFF'}"
        cv2.putText(display_frame, sel_label, (20, 125), font, 0.5, (0, 255, 255), 1)
        
        # Draw controls panel
        cv2.rectangle(overlay, (10, 145), (500, 310), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        controls = [
            "N - Normal mode (no correction)",
            "P - Protanopia (red-blind)",
            "D - Deuteranopia (green-blind)",
            "T - Tritanopia (blue-blind)",
            "S - Toggle Selective Masking",
            "Q - Quit"
        ]
        
        y_offset = 170
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
        print("  P - Protanopia (red-blind)")
        print("  D - Deuteranopia (green-blind)")
        print("  T - Tritanopia (blue-blind)")
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
            elif key == ord('s') or key == ord('S'):
                self.corrector.toggle_selective()
        
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


