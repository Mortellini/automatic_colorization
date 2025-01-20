import pandas as pd
import matplotlib.pyplot as plt

# Pfad zur CSV-Datei mit den Delta-E-Werten
data_path = "results/statistics/delta_e/delta_e_results.csv"

# Ergebnisse laden
results_df = pd.read_csv(data_path)

# Sicherstellen, dass die Spalte "Image" Strings enthält
results_df['Image'] = results_df['Image'].astype(str)

# Scatterplot der Delta-E-Werte für beide Modelle erstellen (mit numerischer X-Achse)
plt.figure(figsize=(12, 6))
for model in results_df['Model'].unique():
    subset = results_df[results_df['Model'] == model]
    indices = range(len(subset))  # Numerierung für die X-Achse
    plt.scatter(indices, subset['Delta E'], label=model, alpha=0.7)

plt.title("Delta-E-Abweichungen der Modelle", fontsize=14)
plt.xlabel("Bildindex", fontsize=12)  # Neue X-Achsenbeschriftung
plt.ylabel("Delta-E-Abweichung", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("results/statistics/delta_e/plots/delta_e_scatterplot_numeric.png")
plt.show()

# Histogramm der Delta-E-Werte erstellen
plt.figure(figsize=(12, 6))
for model in results_df['Model'].unique():
    subset = results_df[results_df['Model'] == model]
    plt.hist(subset['Delta E'], bins=30, alpha=0.5, label=model)

plt.title("Histogramm der Delta-E-Abweichungen", fontsize=14)
plt.xlabel("Delta-E-Abweichung", fontsize=12)
plt.ylabel("Häufigkeit", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("results/statistics/delta_e/plots/delta_e_histogram.png")
plt.show()

# Minimum-, Maximum- und Durchschnittswerte für beide Modelle berechnen
summary_stats = results_df.groupby("Model")['Delta E'].agg(['min', 'max', 'mean']).reset_index()

# Tabelle der Ergebnisse ausgeben
print(summary_stats)

# Tabelle als PNG speichern
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')
ax.table(cellText=summary_stats.values, colLabels=["Model", "Min Delta-E", "Max Delta-E", "Mean Delta-E"], loc='center')
plt.title("Zusammenfassung der Delta-E-Werte")
plt.savefig("results/statistics/delta_e/plots/delta_e_summary_table.png")
plt.show()
