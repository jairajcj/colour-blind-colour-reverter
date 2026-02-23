import cv2
import numpy as np
import os
from colorblind_correction import ColorBlindnessCorrector

def generate_comparison():
    corrector = ColorBlindnessCorrector()
    plates_dir = "ishihara_plates"
    output_dir = "comparison_results"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plates = [f for f in os.listdir(plates_dir) if f.endswith(('.png', '.jpg'))]
    modes = ['protanopia', 'deuteranopia', 'tritanopia']
    
    for plate_file in plates:
        plate_path = os.path.join(plates_dir, plate_file)
        original = cv2.imread(plate_path)
        
        for mode in modes:
            # Apply correction
            corrected = corrector.daltonize(original, mode)
            
            # Combine side-by-side
            comparison = np.hstack((original, corrected))
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(comparison, f"Corrected ({mode})", (original.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
            
            output_path = os.path.join(output_dir, f"cmp_{mode}_{plate_file}")
            cv2.imwrite(output_path, comparison)
            print(f"Generated comparison: {output_path}")

if __name__ == "__main__":
    generate_comparison()
