import os
import cv2
import pandas as pd
import numpy as np
from skimage.color import deltaE_cie76, rgb2lab

# Pfade zu den Verzeichnissen
input_dir = "results/input"
base_model_dir = "results/base_model"
fine_tune_model_dir = "results/fine_tune_model"
output_csv_path = "results/statistics/delta_e/delta_e_results.csv"

# Funktion zur Berechnung der Delta-E-Werte
def calculate_delta_e(image1, image2):
    lab1 = rgb2lab(image1 / 255.0)
    lab2 = rgb2lab(image2 / 255.0)
    return deltaE_cie76(lab1, lab2).mean()

# Initialisierung der Ergebnisse
results = []

# Berechnung für das Grundmodell
print("Berechnung der Delta-E-Werte für das Grundmodell...")
for file_name in os.listdir(input_dir):
    input_path = os.path.join(input_dir, file_name)
    base_path = os.path.join(base_model_dir, file_name.replace('.jpg', '_base.png'))

    if os.path.isfile(input_path) and os.path.isfile(base_path):
        input_image = cv2.imread(input_path)
        base_image = cv2.imread(base_path)

        if input_image is not None and base_image is not None:
            delta_e_value = calculate_delta_e(input_image, base_image)
            results.append({"Image": file_name, "Model": "Base Model", "Delta E": delta_e_value})

# Berechnung für das Fine-Tuning-Modell
print("Berechnung der Delta-E-Werte für das Fine-Tuning-Modell...")
for file_name in os.listdir(input_dir):
    input_path = os.path.join(input_dir, file_name)
    fine_tune_path = os.path.join(fine_tune_model_dir, file_name.replace('.jpg', '_fine_tune.png'))

    if os.path.isfile(input_path) and os.path.isfile(fine_tune_path):
        input_image = cv2.imread(input_path)
        fine_tune_image = cv2.imread(fine_tune_path)

        if input_image is not None and fine_tune_image is not None:
            delta_e_value = calculate_delta_e(input_image, fine_tune_image)
            results.append({"Image": file_name, "Model": "Fine-Tune Model", "Delta E": delta_e_value})

# Ergebnisse in ein DataFrame speichern
results_df = pd.DataFrame(results)
results_df['Image'] = results_df['Image'].astype(str)

# Überprüfen, ob Ergebnisse existieren
if not results_df.empty:
    # Durchschnitt und Median berechnen
    summary = results_df.groupby("Model")["Delta E"].agg(["mean", "median"]).reset_index()
    summary.rename(columns={"mean": "Mean Delta E", "median": "Median Delta E"}, inplace=True)

    # Zusammenführen der Ergebnisse und Zusammenfassung
    final_results = pd.concat([results_df, summary], ignore_index=True)

    # Ergebnisse speichern
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_results.to_csv(output_csv_path, index=False)

    print(f"Delta-E-Ergebnisse wurden gespeichert unter: {output_csv_path}")
else:
    print("Keine gültigen Ergebnisse gefunden. Überprüfen Sie die Eingabebilder.")
