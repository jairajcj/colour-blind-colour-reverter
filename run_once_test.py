import cv2
import numpy as np
from colorblind_correction import ColorBlindnessCorrector

def create_test_image():
    """Create a colorful test image with various colors."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Create color blocks
    colors = [
        (255, 0, 0),      # Blue (BGR)
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 128, 128),  # Gray
        (255, 128, 0),    # Light Blue
        (128, 0, 255),    # Purple
        (0, 255, 128),    # Light Green
        (255, 128, 128),  # Light Blue-Gray
        (128, 255, 128),  # Light Green-Gray
    ]
    
    # Draw color blocks in a grid
    block_width = 200
    block_height = 100
    
    for i, color in enumerate(colors):
        row = i // 3
        col = i % 3
        x1 = col * block_width
        y1 = row * block_height
        x2 = x1 + block_width
        y2 = y1 + block_height
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        
        # Add text label
        label = f"Color {i+1}"
        cv2.putText(img, label, (x1 + 50, y1 + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img

def main():
    print("Running Color Blindness Correction Test...")
    corrector = ColorBlindnessCorrector()
    test_image = create_test_image()
    
    # Apply corrections
    protanopia_corrected = corrector.daltonize(test_image, 'protanopia')
    deuteranopia_corrected = corrector.daltonize(test_image, 'deuteranopia')
    tritanopia_corrected = corrector.daltonize(test_image, 'tritanopia')
    
    # Stack images for comparison
    top_row = np.hstack((test_image, protanopia_corrected))
    bottom_row = np.hstack((deuteranopia_corrected, tritanopia_corrected))
    combined = np.vstack((top_row, bottom_row))
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Protanopia Corrected", (610, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Deuteranopia Corrected", (10, 430), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Tritanopia Corrected", (610, 430), font, 1, (255, 255, 255), 2)
    
    # Save the result
    output_path = "correction_test_result.png"
    cv2.imwrite(output_path, combined)
    print(f"Test completed. Result saved to {output_path}")

if __name__ == "__main__":
    main()
