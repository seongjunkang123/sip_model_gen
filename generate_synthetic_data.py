import json
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np, pandas as pd
from keras.models import load_model
import os

# constants
LATENT_DIM = 128
NUM_CLASSES = 3
AMT = 500  # samples per condition
OUTPUT_DIRECTORY = "./synthetic_data(500)"

# loading csv file to fit the scaler on log-transformed data
dataset_csv = pd.read_csv('../sip_data/res1/combined_data.csv')
data = dataset_csv.iloc[:, 2:].values

# get column names
voc_list = dataset_csv.columns[2:]

# select version 22 for the generator model
version = 22

# load preprocessing params
preprocessing_path = f"gen_model_weights/preprocessing_v{version}.json"
try:
    with open(preprocessing_path, "r") as f:
        preproc = json.load(f)
    use_log = preproc.get("log_transform", False)
except FileNotFoundError:
    print(f"WARNING: {preprocessing_path} not found, assuming no log transform.")
    use_log = False

# fit scaler the same way as training
if use_log:
    data_for_scaler = np.log1p(data)
else:
    data_for_scaler = data

minmaxscaler = MinMaxScaler()
minmaxscaler.fit(data_for_scaler)

# load model
PATH = f"gen_model_weights/gen_v{version}.keras"
generator = load_model(PATH)

conditions = ["asthma", "bronchi", "copd"]

for i, condition in enumerate(conditions):
    results = []
    print(f"creating {condition} synthetic data")

    for amt in range(AMT):
        # one-hot encode condition
        condition_onehot = tf.one_hot([i], depth=NUM_CLASSES)

        # create random latent vector to put into the generator
        latent_vec = tf.random.normal(shape=(1, LATENT_DIM))

        # generate output (in normalized space)
        generated_output_normalized = generator.predict([latent_vec, condition_onehot], verbose=0)

        # inverse transform: MinMaxScaler → then expm1 if log was used
        generated_output = minmaxscaler.inverse_transform(generated_output_normalized)
        if use_log:
            generated_output = np.expm1(generated_output)

        # clip negative values to 0 (VOC concentrations can't be negative)
        generated_output = np.maximum(generated_output, 0)

        # add patient id and condition to row data
        row_data = {
            "Patient ID": f"{condition}_{amt + 1}",
            "Disease": condition,
        }

        # add individual voc profiles to row
        row_data.update(dict(zip(voc_list, generated_output[0])))

        results.append(row_data)

    df = pd.DataFrame(results)
    PATH = f"{OUTPUT_DIRECTORY}/synthetic_data_{condition}.csv"
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.mkdir(OUTPUT_DIRECTORY)
    df.to_csv(PATH, index=False)

    print(f"created and saved {condition} synthetic data ({AMT} samples)")
