import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ordner
input_dir = "results/input"
base_model_dir = "results/base_model"
fine_tune_model_dir = "results/fine_tune_model"
output_dir = "results/color_distributions"
os.makedirs(output_dir, exist_ok=True)

# Funktion: Erzeuge aggregiertes Histogramm
def calculate_histogram(image_files, bins=256):
    r_hist, g_hist, b_hist = np.zeros(bins), np.zeros(bins), np.zeros(bins)
    total_pixels = 0

    for file in image_files:
        img = cv2.imread(file)
        if img is not None:
            total_pixels += img.shape[0] * img.shape[1]
            r_hist += cv2.calcHist([img], [2], None, [bins], [0, 256]).flatten()
            g_hist += cv2.calcHist([img], [1], None, [bins], [0, 256]).flatten()
            b_hist += cv2.calcHist([img], [0], None, [bins], [0, 256]).flatten()

    return r_hist / total_pixels, g_hist / total_pixels, b_hist / total_pixels

# Lade alle Dateien
input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
base_files = [os.path.join(base_model_dir, f) for f in os.listdir(base_model_dir) if f.endswith(('.jpg', '.png'))]
fine_tune_files = [os.path.join(fine_tune_model_dir, f) for f in os.listdir(fine_tune_model_dir) if f.endswith(('.jpg', '.png'))]

# Berechne Histogramme
print("Berechne Farbverteilungen...")
r_input, g_input, b_input = calculate_histogram(input_files)
r_base, g_base, b_base = calculate_histogram(base_files)
r_fine, g_fine, b_fine = calculate_histogram(fine_tune_files)

# Plotten
colors = ["Red", "Green", "Blue"]
input_hists = [r_input, g_input, b_input]
base_hists = [r_base, g_base, b_base]
fine_hists = [r_fine, g_fine, b_fine]

for i, color in enumerate(colors):
    plt.figure(figsize=(10, 6))
    plt.plot(input_hists[i], color="black", label="Ground Truth", linestyle="--")
    plt.plot(base_hists[i], color="blue", label="Base Model", alpha=0.7)
    plt.plot(fine_hists[i], color="red", label="Fine-Tune Model", alpha=0.7)
    plt.title(f"{color} Channel Distribution")
    plt.xlabel("Intensity Value")
    plt.ylabel("Normalized Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{color}_channel_distribution.png"))
    plt.close()

print(f"Plots gespeichert unter {output_dir}")
