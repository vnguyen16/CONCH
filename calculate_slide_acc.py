"""
This script calculates the accuracy per slide for a given CSV file with predictions.
"""
import pandas as pd
import os

# Load CSV with predictions
csv_path = r"C:\Users\Vivian\Documents\CONCH\patch_predictions\patient_split_CONCH70.csv"  # Replace with your file path
df = pd.read_csv(csv_path)

# Extract slide name from 'Patch Path' column
# Assumes structure like: patches\40x\FA\FA 126 B1\patch_XXX.npy
df['Slide'] = df['Patch Path'].apply(lambda x: os.path.normpath(x).split(os.sep)[-2])

# Determine if prediction is correct
df['Correct'] = df['Predicted'] == df['True Label']

# Group by slide and calculate accuracy
accuracy_per_slide = df.groupby('Slide')['Correct'].mean().reset_index()
accuracy_per_slide.columns = ['Slide', 'Accuracy']
accuracy_per_slide['Accuracy (%)'] = accuracy_per_slide['Accuracy'] * 100

# Display or save
print(accuracy_per_slide.sort_values(by='Accuracy (%)', ascending=False))

# Optional: save to CSV
# accuracy_per_slide.to_csv("slide_acc/conch70_patient.csv", index=False)
