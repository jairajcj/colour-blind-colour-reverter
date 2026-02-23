import cv2
import numpy as np
import os
from colorblind_correction import ColorBlindnessCorrector

def create_realistic_scene():
    """Create a realistic multi-color scene simulating real-life objects."""
    w, h = 800, 500
    scene = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Sky gradient (blue to light blue)
    for y in range(200):
        ratio = y / 200.0
        scene[y, :] = [int(255 - 80*ratio), int(200 - 50*ratio), int(50 + 100*ratio)]  # BGR
    
    # Sun
    cv2.circle(scene, (650, 80), 50, (0, 200, 255), -1)  # Yellow sun
    cv2.circle(scene, (650, 80), 60, (0, 180, 255), 3)
    
    # Clouds
    cv2.ellipse(scene, (200, 60), (80, 30), 0, 0, 360, (230, 230, 230), -1)
    cv2.ellipse(scene, (250, 50), (60, 25), 0, 0, 360, (240, 240, 240), -1)
    
    # Green grass field
    for y in range(200, h):
        ratio = (y - 200) / 300.0
        green_val = int(180 - 80*ratio)
        scene[y, :] = [0, green_val, int(30 + 20*ratio)]
    
    # Trees
    for tx in [100, 300, 550]:
        # Trunk (brown)
        cv2.rectangle(scene, (tx-10, 150), (tx+10, 250), (30, 60, 100), -1)
        # Leaves (different greens)
        cv2.circle(scene, (tx, 130), 50, (20, 140, 20), -1)
        cv2.circle(scene, (tx-30, 150), 35, (30, 160, 30), -1)
        cv2.circle(scene, (tx+30, 150), 35, (10, 120, 10), -1)
    
    # Red flowers in grass
    for fx, fy in [(150, 300), (180, 320), (400, 350), (450, 310), (600, 380), (350, 400), (250, 370)]:
        cv2.circle(scene, (fx, fy), 8, (0, 0, 220), -1)
        cv2.circle(scene, (fx, fy), 4, (0, 80, 255), -1)
    
    # Yellow flowers
    for fx, fy in [(500, 330), (520, 360), (130, 380), (700, 340)]:
        cv2.circle(scene, (fx, fy), 7, (0, 200, 240), -1)
        cv2.circle(scene, (fx, fy), 3, (0, 240, 255), -1)
    
    # Path (brown/beige)
    pts = np.array([[350, 500], [450, 500], [420, 350], [400, 250], [380, 250], [360, 350]], np.int32)
    cv2.fillPoly(scene, [pts], (100, 140, 180))
    
    # Traffic light
    cv2.rectangle(scene, (720, 150), (760, 300), (40, 40, 40), -1)
    cv2.rectangle(scene, (735, 300), (745, 400), (60, 60, 60), -1)
    cv2.circle(scene, (740, 180), 15, (0, 0, 220), -1)   # Red
    cv2.circle(scene, (740, 220), 15, (0, 200, 230), -1)  # Yellow
    cv2.circle(scene, (740, 260), 15, (0, 200, 0), -1)    # Green
    
    # Fruits on the ground
    cv2.circle(scene, (80, 420), 18, (0, 140, 255), -1)    # Orange
    cv2.circle(scene, (120, 430), 15, (0, 0, 200), -1)     # Red apple
    cv2.circle(scene, (50, 440), 14, (0, 200, 0), -1)      # Green apple
    cv2.circle(scene, (160, 425), 12, (180, 0, 180), -1)   # Purple grape
    
    # Butterfly
    cv2.ellipse(scene, (480, 180), (20, 12), 30, 0, 360, (200, 100, 0), -1)
    cv2.ellipse(scene, (500, 175), (18, 10), -30, 0, 360, (220, 120, 30), -1)
    
    # Rainbow arc (partially visible)
    colors = [(0, 0, 255), (0, 127, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 127), (200, 0, 200)]
    for i, color in enumerate(colors):
        cv2.ellipse(scene, (400, 0), (350 + i*8, 200 + i*8), 0, 40, 140, color, 4)
    
    return scene


def create_fruit_closeup():
    """Fruits and vegetables - common color confusion scenario."""
    w, h = 800, 500
    scene = np.full((h, w, 3), (240, 235, 230), dtype=np.uint8)  # Light table
    
    # Red apple
    cv2.circle(scene, (150, 250), 80, (20, 30, 200), -1)
    cv2.circle(scene, (150, 250), 80, (10, 10, 160), 3)
    cv2.ellipse(scene, (150, 180), (15, 25), 0, 0, 360, (10, 80, 10), -1)  # leaf
    cv2.putText(scene, "Red Apple", (90, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    
    # Green apple
    cv2.circle(scene, (400, 250), 80, (30, 180, 50), -1)
    cv2.circle(scene, (400, 250), 80, (20, 120, 30), 3)
    cv2.ellipse(scene, (400, 180), (15, 25), 0, 0, 360, (10, 60, 10), -1)
    cv2.putText(scene, "Green Apple", (330, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    
    # Orange
    cv2.circle(scene, (650, 250), 75, (0, 130, 255), -1)
    cv2.circle(scene, (650, 250), 75, (0, 100, 200), 3)
    cv2.putText(scene, "Orange", (600, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    
    # Banana cluster at bottom
    cv2.ellipse(scene, (250, 450), (100, 30), -10, 0, 360, (0, 210, 240), -1)
    cv2.putText(scene, "Banana", (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) if h > 480 else None
    
    # Grapes cluster
    for gx, gy in [(530, 430), (550, 430), (570, 430), (540, 450), (560, 450), (550, 470)]:
        cv2.circle(scene, (gx, gy), 15, (120, 0, 100), -1)
    cv2.putText(scene, "Grapes", (510, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) if h > 480 else None
    
    return scene


def create_traffic_scene():
    """Traffic and road signs - critical safety scenario."""
    w, h = 800, 500
    scene = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Road
    cv2.rectangle(scene, (0, 300), (800, 500), (60, 60, 60), -1)
    # Lane markings
    for x in range(0, 800, 100):
        cv2.rectangle(scene, (x, 390), (x+50, 400), (200, 200, 200), -1)
    
    # Sky
    for y in range(300):
        ratio = y / 300
        scene[y, :] = [int(200 - 100*ratio), int(150 - 50*ratio), int(40 + 60*ratio)]
    
    # Traffic light - RED active
    cv2.rectangle(scene, (80, 80), (140, 250), (30, 30, 30), -1)
    cv2.rectangle(scene, (100, 250), (120, 350), (50, 50, 50), -1)
    cv2.circle(scene, (110, 120), 22, (0, 0, 255), -1)    # RED - ON
    cv2.circle(scene, (110, 170), 22, (0, 60, 60), -1)     # Yellow - dim
    cv2.circle(scene, (110, 220), 22, (0, 40, 0), -1)      # Green - dim
    cv2.putText(scene, "STOP", (70, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Traffic light - GREEN active
    cv2.rectangle(scene, (350, 80), (410, 250), (30, 30, 30), -1)
    cv2.rectangle(scene, (370, 250), (390, 350), (50, 50, 50), -1)
    cv2.circle(scene, (380, 120), 22, (0, 0, 60), -1)      # Red - dim
    cv2.circle(scene, (380, 170), 22, (0, 60, 60), -1)     # Yellow - dim
    cv2.circle(scene, (380, 220), 22, (0, 255, 0), -1)     # GREEN - ON
    cv2.putText(scene, "GO", (360, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Stop sign (octagon)
    cx, cy, r = 600, 150, 50
    angles = [np.pi/8 + i * np.pi/4 for i in range(8)]
    pts = np.array([[int(cx + r*np.cos(a)), int(cy + r*np.sin(a))] for a in angles], np.int32)
    cv2.fillPoly(scene, [pts], (0, 0, 200))
    cv2.putText(scene, "STOP", (570, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Green directional sign
    cv2.rectangle(scene, (500, 220), (750, 280), (0, 120, 0), -1)
    cv2.putText(scene, "Highway Exit 5 ->", (510, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Cars
    cv2.rectangle(scene, (200, 340), (300, 380), (0, 0, 180), -1)  # Red car
    cv2.rectangle(scene, (500, 410), (600, 450), (0, 160, 0), -1)  # Green car
    
    return scene


def generate_all_comparisons():
    corrector = ColorBlindnessCorrector()
    output_dir = "reallife_comparisons"
    os.makedirs(output_dir, exist_ok=True)
    
    scenes = {
        "nature_scene": create_realistic_scene(),
        "fruits_vegetables": create_fruit_closeup(),
        "traffic_safety": create_traffic_scene(),
    }
    
    modes = ['protanopia', 'deuteranopia', 'tritanopia']
    mode_labels = {
        'protanopia': 'Protanopia (Red-Blind)',
        'deuteranopia': 'Deuteranopia (Green-Blind)',
        'tritanopia': 'Tritanopia (Blue-Blind)',
    }
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    for scene_name, original in scenes.items():
        for mode in modes:
            corrected = corrector.daltonize(original, mode)
            
            # Create labeled comparison
            h, w = original.shape[:2]
            canvas = np.zeros((h + 70, w * 2 + 20, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)
            
            # Title
            cv2.putText(canvas, f"Real-Life View: {mode_labels[mode]} Correction", 
                        (20, 35), font, 0.9, (255, 255, 255), 2)
            
            # Place images
            canvas[60:60+h, 0:w] = original
            canvas[60:60+h, w+20:w+20+w] = corrected
            
            # Labels under images
            cv2.putText(canvas, "BEFORE (Original)", (int(w/2)-100, 55), font, 0.6, (100, 180, 255), 2)
            cv2.putText(canvas, "AFTER (Corrected)", (w + 20 + int(w/2)-100, 55), font, 0.6, (100, 255, 100), 2)
            
            # Divider line
            cv2.line(canvas, (w+10, 45), (w+10, h+60), (100, 100, 100), 2)
            
            out_path = os.path.join(output_dir, f"{scene_name}_{mode}.png")
            cv2.imwrite(out_path, canvas)
            print(f"Saved: {out_path}")
    
    # Create master grid: 3 scenes x 3 modes
    print("\nGenerating master grid...")
    cell_w, cell_h = 400, 280
    grid = np.zeros((cell_h * 3 + 80, cell_w * 4 + 30, 3), dtype=np.uint8)
    grid[:] = (20, 20, 20)
    
    # Header
    cv2.putText(grid, "Real-Life Color Blindness Correction - Before & After", (30, 40), font, 1, (255, 255, 255), 2)
    
    # Column headers
    cv2.putText(grid, "Original", (50, 70), font, 0.6, (200, 200, 200), 2)
    for i, mode in enumerate(modes):
        cv2.putText(grid, mode_labels[mode], (cell_w*(i+1) + 20, 70), font, 0.5, (100, 255, 100), 1)
    
    scene_labels = ["Nature Scene", "Fruits & Veggies", "Traffic & Safety"]
    for row, (scene_name, original) in enumerate(scenes.items()):
        y_start = 80 + row * cell_h
        
        # Original
        resized_orig = cv2.resize(original, (cell_w - 10, cell_h - 20))
        grid[y_start+10:y_start+10+cell_h-20, 10:10+cell_w-10] = resized_orig
        cv2.putText(grid, scene_labels[row], (15, y_start + cell_h - 5), font, 0.45, (200, 200, 200), 1)
        
        # Each correction mode
        for col, mode in enumerate(modes):
            corrected = corrector.daltonize(original, mode)
            resized_corr = cv2.resize(corrected, (cell_w - 10, cell_h - 20))
            x_start = (col + 1) * cell_w + 20
            grid[y_start+10:y_start+10+cell_h-20, x_start:x_start+cell_w-10] = resized_corr
    
    cv2.imwrite(os.path.join(output_dir, "master_reallife_grid.png"), grid)
    print(f"Master grid saved to: {output_dir}/master_reallife_grid.png")
    print("Done!")

if __name__ == "__main__":
    generate_all_comparisons()
