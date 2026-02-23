"""
Demo script to test color blindness correction on static images.
Use this if you don't have a camera or want to test with specific images.
"""

import cv2
import numpy as np
from colorblind_correction import ColorBlindnessCorrector


def create_test_image():
    """Create a colorful test image with various colors."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Create color blocks
    colors = [
        (255, 0, 0),      # Blue
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
    """Test color blindness correction on a static image."""
    print("Color Blindness Correction - Image Demo")
    print("=" * 50)
    
    # Create test image
    test_image = create_test_image()
    
    # Initialize corrector
    corrector = ColorBlindnessCorrector()
    
    # Create windows
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Protanopia Corrected', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Deuteranopia Corrected', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Tritanopia Corrected', cv2.WINDOW_NORMAL)
    
    # Apply corrections
    protanopia_corrected = corrector.daltonize(test_image, 'protanopia')
    deuteranopia_corrected = corrector.daltonize(test_image, 'deuteranopia')
    tritanopia_corrected = corrector.daltonize(test_image, 'tritanopia')
    
    # Display all versions
    cv2.imshow('Original', test_image)
    cv2.imshow('Protanopia Corrected', protanopia_corrected)
    cv2.imshow('Deuteranopia Corrected', deuteranopia_corrected)
    cv2.imshow('Tritanopia Corrected', tritanopia_corrected)
    
    print("\nDisplaying original and corrected images.")
    print("Press any key to exit...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
