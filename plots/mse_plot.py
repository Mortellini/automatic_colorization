import pandas as pd
import matplotlib.pyplot as plt
import os

# CSV-Datei laden
csv_file = "results/statistics/mse/mse_results.csv"
output_dir = "results/statistics/mse/plots"

# Sicherstellen, dass das Ausgabe-Verzeichnis existiert
os.makedirs(output_dir, exist_ok=True)

# CSV-Datei einlesen
results_df = pd.read_csv(csv_file)

# Entferne die Zusammenfassungszeile für die detaillierten Plots
summary_row = results_df[results_df['Bildname'] == 'Zusammenfassung']
results_df = results_df[results_df['Bildname'] != 'Zusammenfassung']

# Scatter Plot für MSE-Werte
plt.figure(figsize=(12, 6))
plt.scatter(results_df.index, results_df["MSE_Base_Model"], label="Base Model", alpha=0.7, color='blue')
plt.scatter(results_df.index, results_df["MSE_Fine_Tune_Model"], label="Fine-Tune Model", alpha=0.7, color='red')
plt.xlabel("Bildindex")
plt.ylabel("MSE-Wert")
plt.title("Vergleich der MSE-Werte zwischen Base und Fine-Tune Modellen")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mse_scatter_plot.png"))
plt.close()

# Bar Plot für MSE-Werte
plt.figure(figsize=(12, 6))
width = 0.4
x = range(len(results_df))
plt.bar([i - width / 2 for i in x], results_df["MSE_Base_Model"], width=width, label="Base Model", color='blue')
plt.bar([i + width / 2 for i in x], results_df["MSE_Fine_Tune_Model"], width=width, label="Fine-Tune Model",
        color='red')
plt.xlabel("Bildindex")
plt.ylabel("MSE-Wert")
plt.title("Bar-Plot der MSE-Werte zwischen Base und Fine-Tune Modellen")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mse_bar_plot.png"))
plt.close()

# Zusammenfassung als separaten Plot anzeigen
if not summary_row.empty:
    summary_values = summary_row[["MSE_Base_Model", "MSE_Fine_Tune_Model"]].values.flatten()
    labels = ["Base Model", "Fine-Tune Model"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, summary_values, color=['blue', 'red'], alpha=0.7)

    # Werte über den Balken anzeigen
    for bar, value in zip(bars, summary_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.4f}', ha='center', va='bottom',
                 fontsize=10)

    plt.ylabel("Durchschnittlicher MSE-Wert")
    plt.title("Zusammenfassung der MSE-Werte")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mse_summary_plot.png"))
    plt.close()

print(f"Plots gespeichert im Verzeichnis: {output_dir}")
