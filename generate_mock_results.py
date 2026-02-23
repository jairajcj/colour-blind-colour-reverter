import csv
import random

def generate_mock_data():
    participants = [
        ("Alice", "Protanopia"),
        ("Bob", "Deuteranopia"),
        ("Charlie", "Tritanopia"),
        ("David", "Protanopia"),
        ("Eve", "Deuteranopia"),
        ("Frank", "Tritanopia"),
        ("Grace", "Protanopia"),
        ("Heidi", "Deuteranopia"),
        ("Ivan", "Protanopia"),
        ("Judy", "Deuteranopia"),
        ("Kevin", "Protanopia"),
        ("Mallory", "Deuteranopia"),
        ("Niaj", "Tritanopia"),
        ("Olivia", "Protanopia"),
        ("Peggy", "Deuteranopia"),
        ("Sybil", "Protanopia"),
        ("Trent", "Deuteranopia"),
        ("Victor", "Tritanopia"),
        ("Walter", "Protanopia"),
        ("Xavier", "Deuteranopia")
    ]
    
    plates = ["plate1.png", "plate2.png", "plate3.png"]
    
    results = []
    
    for name, deficiency in participants:
        for plate in plates:
            # Test without correction
            # Plate 1 is easy (control), Plates 2-3 are hard for P/D
            if plate == "plate1.png":
                # Everyone sees plate 1
                prob_no_corr = 0.95
            elif deficiency == "Protanopia" and plate == "plate2.png":
                prob_no_corr = 0.1
            elif deficiency == "Deuteranopia" and plate == "plate3.png":
                prob_no_corr = 0.1
            else:
                prob_no_corr = 0.4 # Mixed difficulty
                
            results.append([name, plate, "False", "Correct" if random.random() < prob_no_corr else "Incorrect"])
            
            # Test with correction
            # Correction should significantly improve results
            if plate == "plate1.png":
                prob_with_corr = 0.98
            else:
                prob_with_corr = 0.85 # High success rate with correction
                
            results.append([name, plate, "True", "Correct" if random.random() < prob_with_corr else "Incorrect"])
            
    with open("all_study_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Participant", "Plate", "Correction", "Result"])
        writer.writerows(results)
    
    print(f"Generated mock results for {len(participants)} participants in all_study_results.csv")

if __name__ == "__main__":
    generate_mock_data()
