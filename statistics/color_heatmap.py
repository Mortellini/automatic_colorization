import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Zielgröße für Resizing
TARGET_SIZE = (1500, 1000)

# Funktion zur Initialisierung von Zählern
def initialize_color_counters():
    return np.zeros(256, dtype=np.int64), np.zeros(256, dtype=np.int64), np.zeros(256, dtype=np.int64)

# Funktion zum Resizen und Laden eines Bilds
def load_and_resize_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize(TARGET_SIZE)
            return np.array(img)
    except Exception as e:
        print(f"Fehler beim Laden von {image_path}: {e}")
        return None

# Funktion zur Verarbeitung eines Bildes
def process_image(image, red_counts, green_counts, blue_counts):
    r_values = image[:, :, 0].flatten()
    g_values = image[:, :, 1].flatten()
    b_values = image[:, :, 2].flatten()

    # Zähle nur Werte im Bereich 10-240
    red_counts[10:241] += np.bincount(r_values, minlength=256)[10:241]
    green_counts[10:241] += np.bincount(g_values, minlength=256)[10:241]
    blue_counts[10:241] += np.bincount(b_values, minlength=256)[10:241]

# Funktion zur Erstellung der Farbverteilungsplots
def plot_color_distribution(red_counts, green_counts, blue_counts, title, output_path):
    plt.figure(figsize=(12, 6))
    x = np.arange(10, 241)
    plt.bar(x - 0.3, red_counts[10:241], width=0.3, color='red', alpha=0.7, label='Red Channel')
    plt.bar(x, green_counts[10:241], width=0.3, color='green', alpha=0.7, label='Green Channel')
    plt.bar(x + 0.3, blue_counts[10:241], width=0.3, color='blue', alpha=0.7, label='Blue Channel')
    plt.title(title)
    plt.xlabel("Pixelintensität")
    plt.ylabel("Häufigkeit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Hauptprogramm
if __name__ == "__main__":
    # Ordnerpfade
    input_dir = "results/input"
    base_model_dir = "results/base_model"
    fine_tune_model_dir = "results/fine_tune_model"

    # Initialisiere Zähler
    red_counts_input, green_counts_input, blue_counts_input = initialize_color_counters()
    red_counts_base, green_counts_base, blue_counts_base = initialize_color_counters()
    red_counts_fine, green_counts_fine, blue_counts_fine = initialize_color_counters()

    # Bilderpfade sammeln
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    base_files = [os.path.join(base_model_dir, f.replace(".jpg", "_base.png")) for f in os.listdir(input_dir)]
    fine_tune_files = [os.path.join(fine_tune_model_dir, f.replace(".jpg", "_fine_tune.png")) for f in os.listdir(input_dir)]

    # Ground-Truth-Bilder verarbeiten
    print("Verarbeite Ground-Truth-Bilder...")
    for file in input_files:
        image = load_and_resize_image(file)
        if image is not None:
            process_image(image, red_counts_input, green_counts_input, blue_counts_input)

    # Base-Model-Bilder verarbeiten
    print("Verarbeite Base-Model-Bilder...")
    for file in base_files:
        image = load_and_resize_image(file)
        if image is not None:
            process_image(image, red_counts_base, green_counts_base, blue_counts_base)

    # Fine-Tune-Model-Bilder verarbeiten
    print("Verarbeite Fine-Tune-Model-Bilder...")
    for file in fine_tune_files:
        image = load_and_resize_image(file)
        if image is not None:
            process_image(image, red_counts_fine, green_counts_fine, blue_counts_fine)

    # Plots erstellen
    os.makedirs("results/statistics/color_distribution", exist_ok=True)
    plot_color_distribution(red_counts_input, green_counts_input, blue_counts_input,
                            "Farbverteilung: Ground Truth (ohne Extremwerte)", "results/statistics/color_distribution/ground_truth.png")
    plot_color_distribution(red_counts_base, green_counts_base, blue_counts_base,
                            "Farbverteilung: Base Model (ohne Extremwerte)", "results/statistics/color_distribution/base_model.png")
    plot_color_distribution(red_counts_fine, green_counts_fine, blue_counts_fine,
                            "Farbverteilung: Fine Tune Model (ohne Extremwerte)", "results/statistics/color_distribution/fine_tune_model.png")

    print("Farbverteilungsplots wurden erfolgreich erstellt.")
