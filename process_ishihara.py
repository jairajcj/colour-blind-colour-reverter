
import cv2
import numpy as np
import os
from colorblind_correction import ColorBlindnessCorrector

def process_ishihara():
    # Load the generated Ishihara image
    img_path = r'C:\Users\admin\.gemini\antigravity\brain\e963dc97-3545-4d42-8c3a-5151096eb2c2\ishihara_plate_test_images_1772128644249.png'
    output_dir = r'C:\Users\admin\Documents\jairajcj_projects\colour-blind-colour-reverter\artifacts'
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        return

    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not decode image")
        return

    corrector = ColorBlindnessCorrector()
    
    # Process for each mode
    modes = ['protanopia', 'deuteranopia', 'tritanopia']
    
    # Save original
    cv2.imwrite(os.path.join(output_dir, 'ishihara_original.png'), img)
    
    for mode in modes:
        # Simulate (what the blind person sees)
        simulated_lms = corrector.simulate_colorblindness(corrector.rgb_to_lms(img), mode)
        simulated_bgr = corrector.lms_to_rgb(simulated_lms)
        cv2.imwrite(os.path.join(output_dir, f'ishihara_sim_{mode}.png'), simulated_bgr)
        
        # Daltonize (the fix from your project)
        corrected = corrector.daltonize(img, mode)
        cv2.imwrite(os.path.join(output_dir, f'ishihara_corrected_{mode}.png'), corrected)
        
        print(f"Generated results for {mode}")

if __name__ == "__main__":
    process_ishihara()
