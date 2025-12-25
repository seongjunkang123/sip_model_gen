import keras
import tensorflow as tf
from keras import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import os

# define global variables (manipulate them)
BATCH_SIZE = 16
EPOCHS = 100

# get model version
gen_model_weights_directory = './gen_model_weights'
sorted_files = sorted(os.listdir(gen_model_weights_directory))

last_index = len(sorted_files) - 1
if last_index < 0:
    v = 1
else:
    file = sorted_files[last_index]
    v = int(file[5]) + 1

# loading dataset (csv file)
dataset_csv = pd.read_csv('../sip_data/res1/combined_data.csv')

# filtering data
data = dataset_csv.iloc[:, 2:].values
label = dataset_csv['Disease'].values

# encodes the disease
# 0: Asthma
# 1: Bronchi
# 2: COPD
label_encoded = LabelEncoder().fit_transform(label)
label_onehot = to_categorical(label_encoded, num_classes=3)

# data normalization (minmaxscaling)
# could do StandardScaler in sklearn
data_normalized = MinMaxScaler().fit_transform(data)
print(data_normalized)

# create tensorflow dataset object + shuffle
dataset_tf = tf.data.Dataset.from_tensor_slices((data_normalized, label_onehot))
dataset = dataset_tf.shuffle(buffer_size=1000).batch(BATCH_SIZE)

# more global variables
NUM_CLASS = 3 # asthma, bronchi, copd
LATENT_DIM = 128
NUM_VOCS = data_normalized.shape[1]
print(NUM_VOCS)

# build generator (manipulate this)
def build_generator():
    latent_vector = layers.Input(shape=(LATENT_DIM, ))
    label = layers.Input(shape=(NUM_CLASS, ))

    # change activation to LeakyReLU
    gen = layers.Concatenate()([latent_vector, label])
    gen = layers.Dense(128, activation='relu')(gen)
    gen = layers.Dense(256, activation='relu')(gen)
    gen = layers.Dense(NUM_VOCS, activation='linear')(gen)

    return Model([latent_vector, label], gen, name=f"gen_v{v}")

# build discriminator (manipulate this)
def build_discriminator():
    voc_profile = layers.Input(shape=(NUM_VOCS, ))
    label = layers.Input(shape=(NUM_CLASS, ))

    dis = layers.Concatenate()([voc_profile, label])
    dis = layers.Dense(256, activation='relu')(dis)
    dis = layers.Dense(128, activation='relu')(dis)
    dis = layers.Dense(1, activation='sigmoid')(dis)

    return Model([voc_profile, label], dis, name=f"dis_v{v}")

generator = build_generator()
discriminator = build_discriminator()
