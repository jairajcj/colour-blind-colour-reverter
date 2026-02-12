import cv2
import numpy as np
import os
import json
import csv
import time
from colorblind_correction import ColorBlindnessCorrector

class IshiharaStudy:
    def __init__(self):
        self.corrector = ColorBlindnessCorrector()
        self.participant_data = {
            "name": "",
            "id": int(time.time()),
            "responses": []
        }
        self.plates_dir = "ishihara_plates"
        if not os.path.exists(self.plates_dir):
            os.makedirs(self.plates_dir)
        if not os.listdir(self.plates_dir):
            self.generate_mock_plates()
            
    def generate_mock_plates(self):
        """Generate synthetic Ishihara-like plates if none exist."""
        print("Generating synthetic Ishihara plates for evaluation...")
        # Plate 1: Red '12' on Green (Total Color Blindness test - everyone should see it)
        plate1 = self._create_synthetic_plate("12", (0, 0, 255), (0, 255, 0))
        cv2.imwrite(os.path.join(self.plates_dir, "plate1.png"), plate1)
        
        # Plate 2: Pink '8' on Blue (Protanopia/Deuteranopia test)
        plate2 = self._create_synthetic_plate("8", (147, 20, 255), (255, 0, 0))
        cv2.imwrite(os.path.join(self.plates_dir, "plate2.png"), plate2)
        
        # Plate 3: Green '6' on Red (Deuteranopia test)
        plate3 = self._create_synthetic_plate("6", (0, 255, 0), (0, 0, 255))
        cv2.imwrite(os.path.join(self.plates_dir, "plate3.png"), plate3)

    def _create_synthetic_plate(self, text, foreground_color, background_color):
        """Creates a noisy plate with dots."""
        size = 500
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Draw background noise
        for _ in range(2000):
            center = (np.random.randint(0, size), np.random.randint(0, size))
            radius = np.random.randint(2, 8)
            color = [max(0, min(255, background_color[i] + np.random.randint(-40, 40))) for i in range(3)]
            cv2.circle(img, center, radius, color, -1)
            
        # Draw text as mask
        mask = np.zeros((size, size), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(mask, text, (100, 350), font, 10, 255, 20)
        
        # Draw foreground dots where mask is 255
        for _ in range(1500):
            center = (np.random.randint(0, size), np.random.randint(0, size))
            if mask[center[1], center[0]] == 255:
                radius = np.random.randint(3, 10)
                color = [max(0, min(255, foreground_color[i] + np.random.randint(-40, 40))) for i in range(3)]
                cv2.circle(img, center, radius, color, -1)
                
        return img

    def run(self):
        print("\n" + "="*50)
        print("Ishihara Plate Study - Quantitative Evaluation")
        print("="*50)
        
        self.participant_data["name"] = input("Enter participant name: ")
        cb_type_input = input("Select deficiency to test (P: Protanopia, D: Deuteranopia, T: Tritanopia): ").lower()
        cb_map = {'p': 'protanopia', 'd': 'deuteranopia', 't': 'tritanopia'}
        cb_type = cb_map.get(cb_type_input, 'protanopia')
        
        plates = [f for f in os.listdir(self.plates_dir) if f.endswith(('.png', '.jpg'))]
        
        for plate_file in plates:
            plate_path = os.path.join(self.plates_dir, plate_file)
            img = cv2.imread(plate_path)
            
            # Correction toggle state
            correction_active = False
            
            while True:
                display_img = img.copy()
                if correction_active:
                    display_img = self.corrector.daltonize(img, cb_type)
                
                # Draw UI info
                cv2.putText(display_img, f"Plate: {plate_file}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_img, f"Correction: {'ON' if correction_active else 'OFF'} (Press 'C' to toggle)", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if correction_active else (0, 0, 255), 2)
                cv2.putText(display_img, "Press 'Y' for Correct, 'N' for Incorrect identification", 
                            (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                cv2.imshow("Ishihara Study", display_img)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c'):
                    correction_active = not correction_active
                elif key == ord('y'):
                    self.participant_data["responses"].append({
                        "plate": plate_file,
                        "correction": correction_active,
                        "result": "Correct"
                    })
                    break
                elif key == ord('n'):
                    self.participant_data["responses"].append({
                        "plate": plate_file,
                        "correction": correction_active,
                        "result": "Incorrect"
                    })
                    break
                elif key == 27: # ESC
                    return

        cv2.destroyAllWindows()
        self.save_results()

    def save_results(self):
        filename = f"study_results_{self.participant_data['id']}.json"
        with open(filename, 'w') as f:
            json.dump(self.participant_data, f, indent=4)
        
        # Also append to CSV for easier analysis
        csv_file = "all_study_results.csv"
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Participant", "Plate", "Correction", "Result"])
            
            for resp in self.participant_data["responses"]:
                writer.writerow([self.participant_data["name"], resp["plate"], resp["correction"], resp["result"]])
                
        print(f"\nResults saved to {filename} and all_study_results.csv")

if __name__ == "__main__":
    study = IshiharaStudy()
    study.run()
