import pandas as pd

def analyze_results():
    try:
        df = pd.read_csv("all_study_results.csv")
    except Exception:
        # Fallback to manual if pandas not installed
        import csv
        rows = []
        with open("all_study_results.csv", 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        df = pd.DataFrame(rows)
    
    # Ensure booleans are handled correctly
    df['Correction'] = df['Correction'].map({'True': True, 'False': False, True: True, False: False})
    
    # Calculate overall accuracy
    summary = df.groupby('Correction')['Result'].value_counts(normalize=True).unstack()
    summary = summary.fillna(0) * 100
    
    print("### Quantitative Study Summary")
    print(f"Total Participants: {df['Participant'].nunique()}")
    print(f"Total Trials: {len(df)}")
    print("\nAccuracy (Correct Identification %):")
    print(summary)
    
    # Per type analysis if we had it (we didn't store deficiency type in CSV, but we can infer or just show overall)
    # Actually my generator didn't put deficiency in CSV, let's just group by plate
    
    plate_summary = df.groupby(['Plate', 'Correction'])['Result'].value_counts(normalize=True).unstack()
    plate_summary = plate_summary.fillna(0) * 100
    print("\nAccuracy by Plate:")
    print(plate_summary)

if __name__ == "__main__":
    analyze_results()

