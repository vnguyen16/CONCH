"""
This script calculates the accuracy per slide for a given CSV file with predictions.
"""
# import pandas as pd
# import os

# def calc_accuracy(csv_path, save_path=None):
#     """
#     Calculate accuracy per slide from a CSV file with predictions.

#     Parameters:
#     csv_path (str): Path to the CSV file containing predictions.
#     save_path (str): Optional path to save the results as a CSV file.

#     Returns:
#     pd.DataFrame: DataFrame containing accuracy per slide.
#     """
#     # Load CSV with predictions
#     df = pd.read_csv(csv_path)

#     # Extract slide name from 'Patch Path' column
#     # Assumes structure like: patches\40x\FA\FA 126 B1\patch_XXX.npy
#     df['Slide'] = df['Patch Path'].apply(lambda x: os.path.normpath(x).split(os.sep)[-2])

#     # Determine if prediction is correct
#     df['Correct'] = df['Predicted'] == df['True Label']

#     # Group by slide and calculate accuracy
#     accuracy_per_slide = df.groupby('Slide')['Correct'].mean().reset_index()
#     accuracy_per_slide.columns = ['Slide', 'Accuracy']
#     accuracy_per_slide['Accuracy (%)'] = accuracy_per_slide['Accuracy'] * 100

#     # Display or save
#     print(accuracy_per_slide.sort_values(by='Accuracy (%)', ascending=False))

#     # Optional: save to CSV
#     accuracy_per_slide.to_csv(save_path, index=False)


# def slide_level_majority_voting(csv_path, output_path=None):
#     # Load patch-level predictions
#     df = pd.read_csv(csv_path)

#     # Extract slide ID from the path (adjust depending on your path structure)
#     df['Slide'] = df['Patch Path'].apply(lambda x: os.path.normpath(x).split(os.sep)[-2])

#     # Perform majority voting for each slide
#     slide_predictions = df.groupby('Slide')['Predicted'].agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
#     slide_true_labels = df.groupby('Slide')['True Label'].agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])

#     # Combine results into DataFrame
#     slide_df = pd.DataFrame({
#         'Slide': slide_predictions.index,
#         'Slide-Level Predicted': slide_predictions.values,
#         'Slide-Level True': slide_true_labels.values
#     })

#     # Optionally save to CSV
#     if output_path:
#         slide_df.to_csv(output_path, index=False)
#         print(f"Saved slide-level predictions to: {output_path}")

#     return slide_df

# from sklearn.metrics import accuracy_score

# def evaluate_slide_predictions(slide_df):
#     acc = accuracy_score(slide_df['Slide-Level True'], slide_df['Slide-Level Predicted'])
#     print(f"Slide-level Accuracy: {acc:.4f}")


# def main():
#     csv_path = r'C:\Users\Vivian\Documents\CONCH\patch_predictions\annotated\CONCH_ann_CL_conch_test.csv'
#     # output_path = 'slide_level_predictions.csv'

#     # slide_df = slide_level_majority_voting(csv_path, output_path)
#     slide_df = slide_level_majority_voting(csv_path)
#     evaluate_slide_predictions(slide_df)


# if __name__ == "__main__":
#     main()

import pandas as pd
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

def calc_accuracy(csv_path, save_path=None):
    """
    Calculate accuracy per slide from a CSV file with predictions.

    Parameters:
    csv_path (str): Path to the CSV file containing predictions.
    save_path (str): Optional path to save the results as a CSV file.

    Returns:
    pd.DataFrame: DataFrame containing accuracy per slide.
    """
    df = pd.read_csv(csv_path)
    df['Slide'] = df['Patch Path'].apply(lambda x: os.path.normpath(x).split(os.sep)[-2])
    df['Correct'] = df['Predicted'] == df['True Label']
    accuracy_per_slide = df.groupby('Slide')['Correct'].mean().reset_index()
    accuracy_per_slide.columns = ['Slide', 'Accuracy']
    accuracy_per_slide['Accuracy (%)'] = accuracy_per_slide['Accuracy'] * 100
    print(accuracy_per_slide.sort_values(by='Accuracy (%)', ascending=False))
    if save_path:
        accuracy_per_slide.to_csv(save_path, index=False)
    return accuracy_per_slide


def slide_level_majority_voting(csv_path, output_path=None):
    df = pd.read_csv(csv_path)
    df['Slide'] = df['Patch Path'].apply(lambda x: os.path.normpath(x).split(os.sep)[-2])
    slide_predictions = df.groupby('Slide')['Predicted'].agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
    slide_true_labels = df.groupby('Slide')['True Label'].agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
    slide_df = pd.DataFrame({
        'Slide': slide_predictions.index,
        'Slide-Level Predicted': slide_predictions.values,
        'Slide-Level True': slide_true_labels.values
    })
    if output_path:
        slide_df.to_csv(output_path, index=False)
        print(f"Saved slide-level predictions to: {output_path}")
    return slide_df


def evaluate_slide_predictions(slide_df):
    y_true = slide_df['Slide-Level True']
    y_pred = slide_df['Slide-Level Predicted']

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n🔍 Slide-level Classification Metrics:")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    print("\nDetailed Per-Class Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def main():
    csv_path = r"C:\Users\Vivian\Documents\CONCH\patch_predictions\2.5x\UNI_linprob_test.csv"
    # Optional: set output_path to save slide-level predictions
    # output_path = 'slide_level_predictions.csv'
    print(csv_path)
    slide_df = slide_level_majority_voting(csv_path)
    evaluate_slide_predictions(slide_df)


if __name__ == "__main__":
    main()
