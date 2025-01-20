import os
from PIL import Image

# Funktion, um ein Bild in Graustufen zu konvertieren und abzuspeichern
def save_grayscale_image(image_path, output_dir):
    try:
        # Bild laden und in Graustufen konvertieren
        img = Image.open(image_path).convert('L')  # 'L' für Graustufen

        # Ausgabepfad erstellen
        img_name = os.path.basename(image_path)
        gray_img_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_gray.png")

        # Graustufenbild speichern
        img.save(gray_img_path)
        print(f"Graustufenbild gespeichert unter: {gray_img_path}")
    except Exception as e:
        print(f"Fehler beim Verarbeiten von {image_path}: {e}")

# Funktion, um alle Bilder in einem Ordner zu verarbeiten
def process_images_in_folder(input_folder, output_folder):
    # Erstelle den Ausgabeordner, falls nicht vorhanden
    os.makedirs(output_folder, exist_ok=True)

    # Gehe durch alle Dateien im Eingabeordner
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Überprüfen, ob es sich um eine Bilddatei handelt
        if os.path.isfile(file_path) and filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            save_grayscale_image(file_path, output_folder)
        else:
            print(f"Übersprungen: {file_path} ist keine unterstützte Bilddatei.")

if __name__ == "__main__":
    # Pfade zu den Eingabe- und Ausgabeordnern
    input_folder = "results/input"  # Ordner mit den Originalbildern
    output_folder = "results/grayscale"  # Ordner für Graustufenbilder

    # Verarbeite alle Bilder im Eingabeordner
    process_images_in_folder(input_folder, output_folder)
