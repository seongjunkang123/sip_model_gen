# sip_model_gen

This repository contains the implementation of a Conditional Generative Adversarial Network (CGAN) designed to generate synthetic Volatile Organic Compound (VOC) profiles for various respiratory diseases, including Asthma, Bronchitis, and COPD. The model is built using TensorFlow and Keras.

## Project Overview

The core of this project is a CTGAN (Conditional GAN) that learns the distribution of VOCs from patient data. By providing a disease label (e.g., 'asthma'), the generator can produce new, realistic VOC profiles that are characteristic of that condition. This can be valuable for augmenting limited datasets, balancing class distributions, and for further research into disease biomarkers.

The repository includes scripts for:
*   Preprocessing and combining raw data from multiple sources.
*   Defining and training the CGAN model.
*   Saving model weights, architecture summaries, and performance metrics.
*   Visualizing training history, such as generator and discriminator loss over epochs.

## Directory Structure
For the scripts to function correctly, they expect a specific parent directory structure. This repository (`sip_model_gen`) and the data directory (`sip_data`) must be siblings.

```
/path/to/your/project/
├── sip_data/
│   └── res1/
│       ├── Asthma_peaktable_ver3.csv
│       ├── Bronchi_peaktable_ver3.csv
│       ├── COPD_peaktable_ver3.csv
│       └── intersection_of_detected_compunds.xlsx
├── sip_model_gen/  (This Repository)
│   ├── main.py
│   ├── combine_data.py
│   ├── Dockerfile
│   └── ...
├── sip_model_class/ (Classification Model)
```

## Local Installation and Usage

### Prerequisites
*   Python 3.11.9
*   The directory structure outlined above.
*   The required raw data files placed in the `sip_data/res1/` directory.

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/seongjunkang123/sip_model_gen.git
    cd sip_model_gen
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

Follow these steps in order to prepare the data, train the model, and view the results.

#### 1. Combine and Preprocess Data
Run the `combine_data.py` script to merge the individual disease CSV files into a single dataset for training. This script reads from `../sip_data/res1/`, processes the files, and saves `combined_data.csv` in the same directory.

```bash
python combine_data.py
```

#### 2. Train the Model
The `main.py` script is the entry point for training the GAN. It loads the `combined_data.csv`, builds the generator and discriminator, and runs the training loop.

```bash
python main.py
```

During training, the script will:
*   Automatically version the output files (e.g., `gen_v1`, `gen_v2`).
*   Save model architecture summaries to `gen_model_info/` and `dis_model_info/`.
*   Save the best model weights to `gan_model_weights/`.
*   Save the final generator and discriminator weights to `gen_model_weights/` and `dis_model_weights/`.
*   Save the training loss history to `gan_model_performance/`.

#### 3. Visualize Training History
After training is complete, use the `history.py` script to generate a plot of the generator and discriminator loss curves. This helps in assessing the training stability and performance.

```bash
# The script defaults to the latest version, or you can modify the VERSION variable inside it.
python history.py
```
This will save a `.png` plot in the `gan_model_performance` directory and display it.

## Docker Usage

The provided `Dockerfile` allows you to build a container with all necessary dependencies to run the project in an isolated environment.

### 1. Build the Docker Image
Navigate to the root of this repository (`sip_model_gen`) and run the following command:

```bash
docker build -t sip-model-gen .
```

### 2. Run the Container
Because the scripts use relative paths (`../sip_data`) to access the data, you must mount the parent directory of both `sip_model_gen` and `sip_data` into the container and set the correct working directory.

From the `sip_model_gen` directory, run the command for the desired script.

**To run data preprocessing:**
```bash
docker run --rm -v "$(pwd)/..:/workspace" -w /workspace/sip_model_gen sip-model-gen python combine_data.py
```

**To run model training:**
```bash
docker run --rm -v "$(pwd)/..:/workspace" -w /workspace/sip_model_gen sip-model-gen python main.py
```

These commands map your project's parent directory to `/workspace` inside the container, set the working directory to `/workspace/sip_model_gen`, and then execute the Python script. All generated artifacts (models, plots, etc.) will be saved to your local file system.


## Key Scripts

*   `main.py`: The main script for training the model. It handles data loading and normalization, model definition, training execution via the `CTGAN` class, and saving of all resulting model artifacts.
*   `CTGAN.py`: Defines a custom Keras `Model` subclass that wraps the generator and discriminator. It contains the core `train_step` logic for a conditional GAN, managing the separate optimization steps for both networks.
*   `combine_data.py`: A preprocessing utility that loads raw VOC data for Asthma, Bronchitis, and COPD, finds the common compounds, and merges them into a single `combined_data.csv` file, which serves as the input for `main.py`.
*   `history.py`: A utility to visualize the model's training performance. It loads the pickled history file created during training and plots the discriminator and generator losses over epochs.
*   `utils.py`: Contains helper functions, primarily `get_version()`, which scans the model weights directory to determine the next sequential version number for saving new model artifacts.
*   `generate.py`: A placeholder script intended for using a trained generator to create and save new synthetic VOC profiles.