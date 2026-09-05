import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
from keras.models import load_model

# constants
LATENT_DIM = 128
NUM_CLASSES = 3

# loading csv file to fit the scaler on log-transformed data
dataset_csv = pd.read_csv('../sip_data/res1/combined_data.csv')
data = dataset_csv.iloc[:, 2:].values

# get version and condition
version = input('version: ')

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

print(f"1: asthma\n2: bronchi\n3: copd")
condition = int(input('label: '))
condition_index = condition - 1

# one-hot encode condition
condition_onehot = tf.one_hot([condition_index], depth=NUM_CLASSES)

# load model
PATH = f"gen_model_weights/gen_v{version}.keras"
generator = load_model(PATH)

# create random latent vector to put into the generator
latent_vec = tf.random.normal(shape=(1, LATENT_DIM))

# generate output
generated_output_normalized = generator.predict([latent_vec, condition_onehot])

# inverse transform
generated_output = minmaxscaler.inverse_transform(generated_output_normalized)
if use_log:
    generated_output = np.expm1(generated_output)

generated_output = np.maximum(generated_output, 0)

print(generated_output)
