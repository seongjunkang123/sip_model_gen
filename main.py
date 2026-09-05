import os, datetime, pandas as pd, keras, pickle, json
import numpy as np
import tensorflow as tf
from keras import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from CTGAN import CTGAN
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from convert_to_spreadsheet import add_latest_to_spreadsheet
from utils import get_version

# constants
BATCH_SIZE = 32
EPOCHS = 2000       # more epochs needed for WGAN-GP to converge
LATENT_DIM = 128
NUM_CLASS = 3       # asthma, bronchi, copd
N_CRITIC = 5        # discriminator updates per generator update
GP_WEIGHT = 10.0    # gradient penalty weight
LR = 0.0001         # lower LR for WGAN-GP stability
BETA_1 = 0.0
BETA_2 = 0.9

# get model version
v = get_version()

# load dataset
dataset_csv = pd.read_csv('../sip_data/res1/combined_data.csv')

# filtering data
data = dataset_csv.iloc[:, 2:].values
label = dataset_csv['Disease'].values

# preprocessing: log1p transform + MinMaxScaler
data_log = np.log1p(data)

minmaxscaler = MinMaxScaler()
data_normalized = minmaxscaler.fit_transform(data_log)

# save preprocessing params so generate script can reverse them
preprocessing_params = {
    "log_transform": True,
    "scaler_min": minmaxscaler.data_min_.tolist(),
    "scaler_max": minmaxscaler.data_max_.tolist(),
    "scaler_scale": minmaxscaler.scale_.tolist(),
}
with open(f"./gen_model_weights/preprocessing_v{v}.json", "w") as f:
    json.dump(preprocessing_params, f)

# encode labels
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(label)
label_onehot = to_categorical(label_encoded, num_classes=NUM_CLASS)

# create tensorflow object dataset
dataset_tf = tf.data.Dataset.from_tensor_slices(
    (data_normalized.astype(np.float32), label_onehot.astype(np.float32))
)
dataset = dataset_tf.shuffle(buffer_size=1000).batch(BATCH_SIZE)

NUM_VOCS = data_normalized.shape[1]
print(f"Number of VOC features: {NUM_VOCS}")
print(f"Training samples: {data_normalized.shape[0]}")
print(f"Using WGAN-GP with n_critic={N_CRITIC}, gp_weight={GP_WEIGHT}")


# build generator
def build_generator():
    latent_vector = layers.Input(shape=(LATENT_DIM,))
    label = layers.Input(shape=(NUM_CLASS,))

    gen = layers.Concatenate()([latent_vector, label])

    gen = layers.Dense(256)(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.BatchNormalization()(gen)

    gen = layers.Dense(512)(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.BatchNormalization()(gen)

    gen = layers.Dense(512)(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.BatchNormalization()(gen)

    gen = layers.Dense(256)(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.BatchNormalization()(gen)

    gen = layers.Dense(NUM_VOCS, activation='sigmoid')(gen)

    return Model([latent_vector, label], gen, name=f"gen_v{v}")


# build discriminator
def build_discriminator():
    voc_profile = layers.Input(shape=(NUM_VOCS,))
    label = layers.Input(shape=(NUM_CLASS,))

    dis = layers.Concatenate()([voc_profile, label])

    dis = layers.Dense(512)(dis)
    dis = layers.LeakyReLU(alpha=0.2)(dis)
    dis = layers.LayerNormalization()(dis)
    dis = layers.Dropout(0.3)(dis)

    dis = layers.Dense(256)(dis)
    dis = layers.LeakyReLU(alpha=0.2)(dis)
    dis = layers.LayerNormalization()(dis)
    dis = layers.Dropout(0.3)(dis)

    dis = layers.Dense(128)(dis)
    dis = layers.LeakyReLU(alpha=0.2)(dis)
    dis = layers.LayerNormalization()(dis)
    dis = layers.Dropout(0.3)(dis)

    dis = layers.Dense(64)(dis)
    dis = layers.LeakyReLU(alpha=0.2)(dis)

    # WGAN-GP: no sigmoid on discriminator output (linear activation)
    dis = layers.Dense(1)(dis)

    return Model([voc_profile, label], dis, name=f"dis_v{v}")


generator = build_generator()
discriminator = build_discriminator()

# define paths
GEN_WEIGHTS_PATH = f"./gen_model_weights/gen_v{v}.keras"
GEN_INFO_PATH = f"./gen_model_info/gen_v{v}.txt"

DIS_WEIGHTS_PATH = f"./dis_model_weights/dis_v{v}.keras"
DIS_INFO_PATH = f"./dis_model_info/dis_v{v}.txt"

GAN_WEIGHTS_PATH = f"./gan_model_weights/gan_v{v}.keras"
GAN_PERF_PATH = f"./gan_model_performance/gan_v{v}.pkl"

# save model information
with open(GEN_INFO_PATH, 'w') as f:
    f.write(f"gen_v{v} info\n\n")
    f.write(f"Architecture: Dense-only (no LSTM)\n")
    f.write(f"Loss: Wasserstein + Gradient Penalty\n")
    f.write(f"Preprocessing: log1p + MinMaxScaler\n")
    f.write(f"Latent Dimension: {LATENT_DIM}\n")
    f.write(f"Number of Classes: {NUM_CLASS}\n")
    f.write(f"Number of VOCs: {NUM_VOCS}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"N_Critic: {N_CRITIC}\n")
    f.write(f"GP Weight: {GP_WEIGHT}\n")
    f.write(f"Learning Rate: {LR}\n\n")
    generator.summary(print_fn=lambda x: f.write(x + '\n'))

with open(DIS_INFO_PATH, 'w') as f:
    f.write(f"dis_v{v} info\n\n")
    discriminator.summary(print_fn=lambda x: f.write(x + '\n'))

# make CTGAN
gan = CTGAN(
    discriminator=discriminator,
    generator=generator,
    latent_dim=LATENT_DIM,
    num_classes=NUM_CLASS,
    n_critic=N_CRITIC,
    gp_weight=GP_WEIGHT,
)

# Adam optimizer with beta_1=0, beta_2=0.9
gan.compile(
    g_optimizer=Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2),
    d_optimizer=Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2),
)

# early stopping callbacks
early_stopping_callback = EarlyStopping(
    monitor="g_loss",
    patience=200,
    restore_best_weights=True,
)

# train model
history = gan.fit(dataset, epochs=EPOCHS,)

# save weights
discriminator.save(DIS_WEIGHTS_PATH)
generator.save(GEN_WEIGHTS_PATH)

# save performance
with open(GAN_PERF_PATH, 'wb') as f:
    pickle.dump(history.history, f)

add_latest_to_spreadsheet(v)
