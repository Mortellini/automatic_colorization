# Automatic Colorization (Application for Bachelor Thesis)
Automatic Colorization builds upon the foundational work of Zhang et al. (https://github.com/richzhang/colorization), enhancing their model by implementing and training a fine-tune layer. This adaptation refines the model using a dataset tailored to the city of Hamburg, allowing for more context-aware and accurate colorization of historical images.
## Project Structure

- `colorizer.py`: Main loop for colorizing pictures.
- `gray.py`: Export of gray scale images
- `model/`: Contains the model definition and training loop.
- `statistics/`: Methods for various statistical calculations.
- `plots/`: Methods for plotting statistical results.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Mortellini/automatic_colorization.git
   cd automatic_colorization
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pretrained models:
   - Original Model by Zhang https://github.com/richzhang/colorization (download link in there)
   - [Fine-Tune Model Link](#) https://drive.google.com/file/d/179ZON0usUN4qQlyjhUsEx5sZj3tD6dZE/view?usp=drive_link
   - Place the downloaded model files in the `models/` directory.
   - It should be possible to run the programm without the fine-tune model. It should then create a new fine-tune model file

## Usage

To colorize images using the provided script:

1. **Prepare Input Images**:
   - Place the black-and-white images to be colorized in a directory, e.g., `input_images/`.

2. **Run the Script**:
   - Use the `colorizer.py` script with the required input and output directories:
     ```bash
     python colorizer.py -i input_images/ -o output_images/
     ```

3. **Outputs**:
   - The processed images will be saved in the specified `output_images/` directory, organized into three subdirectories:
     - `base_model/`: Colorized images using the base pretrained model.
     - `fine_tune_model/`: Colorized images using the fine-tuned model.


4. **File Format**:
   - The output images will be named based on the original filenames with additional suffixes (e.g., `example_base.png`, `example_fine_tune.png`).

### Example

If your input images are in a folder named `input_images/`, and you want the outputs in `output_images/`:

```bash
python colorizer.py -i input_images/ -o output_images/
```

After running, you will find the results organized in the `output_images/` directory as described above.

### Notes

- Ensure all input images are in `.png`, `.jpg`, or `.jpeg` format.
- The script automatically skips non-image files.

### Important
- Depending if you are using or training the model you have to change the import of basecolor in eccv16.py
