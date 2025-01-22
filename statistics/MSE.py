import os
import cv2
import numpy as np
import pandas as pd

# Verzeichnisse definieren
input_dir = ""
base_model_dir = ""
fine_tune_model_dir = ""
output_dir = ""

# Sicherstellen, dass das Ausgabe-Verzeichnis existiert
os.makedirs(output_dir, exist_ok=True)

# Liste aller Eingabebilder abrufen
input_images = sorted(os.listdir(input_dir))
base_model_images = sorted(os.listdir(base_model_dir))
fine_tune_model_images = sorted(os.listdir(fine_tune_model_dir))

# Überprüfen, ob alle Verzeichnisse gleich viele Bilder enthalten
assert len(input_images) == len(base_model_images) == len(fine_tune_model_images), "Die Anzahl der Bilder stimmt nicht überein."

# Ergebnisse speichern
results = []

for i, img_name in enumerate(input_images):
    # Lade die Ground Truth, Base Model und Fine-Tune Bilder
    input_img = cv2.imread(os.path.join(input_dir, img_name), cv2.IMREAD_COLOR)
    base_img = cv2.imread(os.path.join(base_model_dir, base_model_images[i]), cv2.IMREAD_COLOR)
    fine_tune_img = cv2.imread(os.path.join(fine_tune_model_dir, fine_tune_model_images[i]), cv2.IMREAD_COLOR)

    # Überprüfen, ob die Bilder erfolgreich geladen wurden
    if input_img is None or base_img is None or fine_tune_img is None:
        print(f"Fehler beim Laden der Bilder: {img_name}")
        continue

    # Berechne MSE für Base Model und Fine-Tune Model
    mse_base = np.mean((input_img - base_img) ** 2)
    mse_fine_tune = np.mean((input_img - fine_tune_img) ** 2)

    # Ergebnisse speichern
    results.append({
        "Bildname": img_name,
        "MSE_Base_Model": mse_base,
        "MSE_Fine_Tune_Model": mse_fine_tune
    })

# Daten in ein DataFrame laden
results_df = pd.DataFrame(results)

# Median und Durchschnitt hinzufügen
summary = {
    "Bildname": "Zusammenfassung",
    "MSE_Base_Model": results_df["MSE_Base_Model"].mean(),
    "MSE_Fine_Tune_Model": results_df["MSE_Fine_Tune_Model"].mean()
}
results_df = pd.concat([results_df, pd.DataFrame([summary])], ignore_index=True)

# CSV speichern
results_df.to_csv(os.path.join(output_dir, "mse_results.csv"), index=False)

print("MSE-Berechnung abgeschlossen. Ergebnisse gespeichert unter:", os.path.join(output_dir, "mse_results.csv"))
