import cv2
import numpy as np
import os
from colorblind_correction import ColorBlindnessCorrector

def create_master_comparison():
    corrector = ColorBlindnessCorrector()
    plates_dir = "ishihara_plates"
    output_file = "master_ishihara_comparison.png"
    
    # We'll use 3 plates and 1 correction type for the main showcase
    plates = ["plate1.png", "plate2.png", "plate3.png"]
    mode = 'protanopia' # Most common test
    
    rows = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for plate_name in plates:
        plate_path = os.path.join(plates_dir, plate_name)
        if not os.path.exists(plate_path):
            continue
            
        original = cv2.imread(plate_path)
        # Resize for consistent grid if needed, though our generator makes them 500x500
        original = cv2.resize(original, (400, 400))
        
        # Apply correction
        corrected = corrector.daltonize(original, mode)
        
        # Add labels to individual images before stacking
        cv2.putText(original, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(corrected, f"Corrected", (10, 30), font, 0.8, (255, 255, 255), 2)
        
        # Combine side-by-side
        row = np.hstack((original, corrected))
        rows.append(row)
    
    if not rows:
        print("No plates found to compare.")
        return
        
    # Stack all rows vertically
    master = np.vstack(rows)
    
    # Add a header
    header = np.zeros((60, master.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, f"Ishihara Plate Study: Protanopia Correction Results", (20, 40), font, 1, (255, 255, 255), 2)
    
    final_image = np.vstack((header, master))
    
    cv2.imwrite(output_file, final_image)
    print(f"Master comparison saved to: {output_file}")

if __name__ == "__main__":
    create_master_comparison()
