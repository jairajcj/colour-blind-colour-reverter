import cv2
import numpy as np
import time
from colorblind_correction import ColorBlindnessCorrector

def run_benchmark():
    corrector = ColorBlindnessCorrector()
    resolutions = [
        (640, 480),   # 480p SD
        (1280, 720),  # 720p HD
        (1920, 1080)  # 1080p Full HD
    ]
    modes = ['protanopia', 'deuteranopia', 'tritanopia']
    iterations = 30  # Number of frames to process for average
    
    print("="*60)
    print(f"{'Resolution':<15} | {'Mode':<15} | {'Avg Time (ms)':<15} | {'FPS':<10}")
    print("-" * 60)
    
    for res in resolutions:
        w, h = res
        # Create a dummy frame
        frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        
        for mode in modes:
            # Warm up
            corrector.daltonize(frame, mode)
            
            start_time = time.time()
            for _ in range(iterations):
                corrector.daltonize(frame, mode)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / iterations * 1000
            fps = 1000 / avg_time
            
            print(f"{f'{w}x{h}':<15} | {mode:<15} | {avg_time:<15.2f} | {fps:<10.2f}")
    
    print("="*60)
    print("Benchmark complete.")

if __name__ == "__main__":
    run_benchmark()
