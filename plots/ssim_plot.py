import os
import pandas as pd
import matplotlib.pyplot as plt

# Pfad zur CSV-Datei
input_csv_path = "results/statistics/ssim/ssim_results.csv"
output_plot_dir = "results/statistics/plots/"

# Erstelle das Ausgabe-Verzeichnis, falls es nicht existiert
os.makedirs(output_plot_dir, exist_ok=True)

# CSV-Datei laden
results_df = pd.read_csv(input_csv_path)

# Entferne die Zeile mit den Zusammenfassungswerten, falls vorhanden
summary_row = results_df[results_df['Image'] == 'Summary']
if not summary_row.empty:
    summary_mean_base = summary_row['SSIM_Base_Model'].values[0]
    summary_mean_fine_tune = summary_row['SSIM_Fine_Tune_Model'].values[0]
    results_df = results_df[results_df['Image'] != 'Summary']
else:
    summary_mean_base = None
    summary_mean_fine_tune = None

# Minimal- und Maximalwerte finden
min_base = results_df.loc[results_df['SSIM_Base_Model'].idxmin()]
max_base = results_df.loc[results_df['SSIM_Base_Model'].idxmax()]
min_fine_tune = results_df.loc[results_df['SSIM_Fine_Tune_Model'].idxmin()]
max_fine_tune = results_df.loc[results_df['SSIM_Fine_Tune_Model'].idxmax()]

# SSIM-Werte des anderen Modells für die Extremfälle
min_base_in_fine_tune = results_df.loc[results_df['Image'] == min_base['Image'], 'SSIM_Fine_Tune_Model'].values[0]
max_base_in_fine_tune = results_df.loc[results_df['Image'] == max_base['Image'], 'SSIM_Fine_Tune_Model'].values[0]
min_fine_tune_in_base = results_df.loc[results_df['Image'] == min_fine_tune['Image'], 'SSIM_Base_Model'].values[0]
max_fine_tune_in_base = results_df.loc[results_df['Image'] == max_fine_tune['Image'], 'SSIM_Base_Model'].values[0]

# Ausgabe der Extremwerte
print("Extremwerte:")
print(f"Base Model - Minimum: {min_base['SSIM_Base_Model']} (Image: {min_base['Image']}), Corresponding in Fine-Tune: {min_base_in_fine_tune}")
print(f"Base Model - Maximum: {max_base['SSIM_Base_Model']} (Image: {max_base['Image']}), Corresponding in Fine-Tune: {max_base_in_fine_tune}")
print(f"Fine-Tune Model - Minimum: {min_fine_tune['SSIM_Fine_Tune_Model']} (Image: {min_fine_tune['Image']}), Corresponding in Base: {min_fine_tune_in_base}")
print(f"Fine-Tune Model - Maximum: {max_fine_tune['SSIM_Fine_Tune_Model']} (Image: {max_fine_tune['Image']}), Corresponding in Base: {max_fine_tune_in_base}")

# Histogram mit angepassten Farben und kleineren Bins
plt.figure(figsize=(10, 6))
plt.hist(results_df['SSIM_Base_Model'], bins=50, alpha=0.7, label='Base Model', color='blue', density=True)
plt.hist(results_df['SSIM_Fine_Tune_Model'], bins=50, alpha=0.7, label='Fine-Tune Model', color='red', density=True)
plt.xlabel('SSIM')
plt.ylabel('Density')
plt.title('Density Distribution of SSIM Values')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_plot_dir, 'ssim_density_overlay.png'))
plt.close()

# Barplot mit feiner Achsenskalierung und Annotationen
plt.figure(figsize=(8, 6))
models = ['Base Model', 'Fine-Tune Model']
means = [summary_mean_base, summary_mean_fine_tune]
y_pos = range(len(models))
plt.bar(models, means, color=['blue', 'red'], alpha=0.7)
plt.ylim(0.6, 1.0)  # Y-Achse auf relevante Werte beschränken
for i, mean in enumerate(means):
    plt.text(i, mean + 0.005, f'{mean:.4f}', ha='center', va='bottom')
plt.xlabel('Model')
plt.ylabel('Mean SSIM')
plt.title('Mean SSIM Comparison Between Models')
plt.tight_layout()
plt.savefig(os.path.join(output_plot_dir, 'ssim_mean_comparison_annotated.png'))
plt.close()

print(f"Plots wurden gespeichert unter: {output_plot_dir}")
