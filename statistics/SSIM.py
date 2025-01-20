import os
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# Ordnerpfade
input_dir = "hk/input"
base_model_dir = "hk/base_model"
fine_tune_model_dir = "hk/fine_tune_model"
output_csv_path = "hk/statistics/ssim/ssim_results.csv"

# Suffixe für die Modelle
base_suffix = "_base"
fine_tune_suffix = "_fine_tune"

# Erstelle das Ausgabe-Verzeichnis, falls es nicht existiert
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

# Funktion zur Berechnung des SSIM
def calculate_ssim(image1, image2):
    # Setze win_size auf die maximale ungerade Zahl kleiner oder gleich der kleinsten Bilddimension
    min_dim = min(image1.shape[0], image1.shape[1])
    win_size = min(7, min_dim if min_dim % 2 != 0 else min_dim - 1)
    return ssim(image1, image2, win_size=win_size, channel_axis=-1)

# Funktion zur Suche einer Datei mit unterschiedlicher Erweiterung
def find_file_with_different_extension(directory, base_name):
    for ext in [".jpg", ".png", ".jpeg"]:
        file_path = os.path.join(directory, base_name + ext)
        if os.path.isfile(file_path):
            return file_path
    return None

# Ergebnisse speichern
results = []

print("Berechnung des SSIM für alle Bilder...")
for file_name in os.listdir(input_dir):
    # Basisname der Datei ohne Erweiterung
    base_name, _ = os.path.splitext(file_name)

    # Input-Pfad
    input_path = os.path.join(input_dir, file_name)

    # Suche die entsprechenden Dateien in den anderen Verzeichnissen
    base_path = find_file_with_different_extension(base_model_dir, base_name + base_suffix)
    fine_tune_path = find_file_with_different_extension(fine_tune_model_dir, base_name + fine_tune_suffix)

    # Überprüfe, ob die entsprechenden Dateien existieren
    if os.path.isfile(input_path) and base_path and fine_tune_path:
        # Bilder laden
        input_image = cv2.imread(input_path)
        base_image = cv2.imread(base_path)
        fine_tune_image = cv2.imread(fine_tune_path)

        # Überprüfe, ob die Bilder geladen wurden und die gleiche Größe haben
        if input_image is not None and base_image is not None and fine_tune_image is not None:
            if input_image.shape == base_image.shape == fine_tune_image.shape:
                # SSIM für Base Model berechnen
                ssim_base = calculate_ssim(input_image, base_image)
                # SSIM für Fine-Tune Model berechnen
                ssim_fine_tune = calculate_ssim(input_image, fine_tune_image)

                # Ergebnisse speichern
                results.append({
                    "Image": file_name,
                    "SSIM_Base_Model": ssim_base,
                    "SSIM_Fine_Tune_Model": ssim_fine_tune
                })
            else:
                print(f"Bildgrößen stimmen nicht überein für {file_name}. Übersprungen.")
        else:
            print(f"Fehler beim Laden der Bilder für {file_name}. Übersprungen.")
    else:
        print(f"Fehlende Dateien für {file_name}. Übersprungen.")

# Ergebnisse in ein DataFrame speichern
results_df = pd.DataFrame(results)

# Überprüfen, ob Ergebnisse existieren
if not results_df.empty:
    # Durchschnitt und Median berechnen
    summary = {
        "Image": "Summary",
        "SSIM_Base_Model": results_df["SSIM_Base_Model"].mean(),
        "SSIM_Fine_Tune_Model": results_df["SSIM_Fine_Tune_Model"].mean()
    }
    results_df = pd.concat([results_df, pd.DataFrame([summary])], ignore_index=True)

    # Ergebnisse speichern
    results_df.to_csv(output_csv_path, index=False)
    print(f"SSIM-Ergebnisse wurden gespeichert unter: {output_csv_path}")
else:
    print("Keine gültigen Ergebnisse gefunden. Überprüfen Sie die Eingabebilder.")
