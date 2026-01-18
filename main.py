import os, datetime, pandas as pd, keras, pickle
import tensorflow as tf
from keras import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from CTGAN import CTGAN
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from utils import get_version

# define global variables (manipulate them)
BATCH_SIZE = 16
EPOCHS = 100

# get model version
v = get_version()

# loading dataset (csv file)
dataset_csv = pd.read_csv('../sip_data/res1/combined_data.csv')

# filtering data
data = dataset_csv.iloc[:, 2:].values
label = dataset_csv['Disease'].values

# encodes the disease
# 0: Asthma
# 1: Bronchi
# 2: COPD
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(label)
label_onehot = to_categorical(label_encoded, num_classes=3)

# data normalization (minmaxscaling)
# could do StandardScaler in sklearn
minmaxscaler = MinMaxScaler()
data_normalized = minmaxscaler.fit_transform(data)
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
    gen = layers.Dense(128)(gen)
    gen = layers.LeakyReLU(alpha=0.4)(gen)
    gen = layers.Dropout(0.5)(gen)
    gen = layers.Dense(256)(gen)
    gen = layers.LeakyReLU(alpha=0.4)(gen)
    gen = layers.Dropout(0.5)(gen)
    gen = layers.Dense(NUM_VOCS, activation='linear')(gen)

    return Model([latent_vector, label], gen, name=f"gen_v{v}")

# build discriminator (manipulate this)
def build_discriminator():
    voc_profile = layers.Input(shape=(NUM_VOCS, ))
    label = layers.Input(shape=(NUM_CLASS, ))

    dis = layers.Concatenate()([voc_profile, label])
    dis = layers.Dense(256)(dis)
    dis = layers.LeakyReLU(alpha=0.4)(dis)
    dis = layers.Dense(128)(dis)
    dis = layers.LeakyReLU(alpha=0.4)(dis)
    dis = layers.Dense(1, activation='sigmoid')(dis)

    return Model([voc_profile, label], dis, name=f"dis_v{v}")

generator = build_generator()
discriminator = build_discriminator()

# define paths for models
GEN_WEIGHTS_PATH    = f"./gen_model_weights/gen_v{v}.keras"
GEN_INFO_PATH       = f"./gen_model_info/gen_v{v}.txt"
GEN_INFO_IMG_PATH   = f"./gen_model_info/gen_v{v}.png"

DIS_WEIGHTS_PATH    = f"./dis_model_weights/gen_v{v}.keras"
DIS_INFO_PATH       = f"./dis_model_info/gen_v{v}.txt"
DIS_INFO_IMG_PATH   = f"./dis_model_info/gen_v{v}.png"

GAN_WEIGHTS_PATH    = f"./gan_model_weights/gan_v{v}.keras"
GAN_PERF_PATH       = f"./gan_model_performance/gan_v{v}.pkl"

# save model information
with open(GEN_INFO_PATH, 'w') as f:
    f.write(f"gen_v{v} info\n\n")

    f.write(f"Latent Dimension: {LATENT_DIM}\n")
    f.write(f"Number of Classes: {NUM_CLASS}\n")
    f.write(f"Number of VOCs: {NUM_VOCS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n\n")

    generator.summary(print_fn=lambda x: f.write(x + '\n'))

with open(DIS_INFO_PATH, 'w') as f:
    f.write(f"dis_v{v} info\n\n")
    discriminator.summary(print_fn=lambda x: f.write(x + '\n'))

# save model info as image
# plot_model(generator, to_file=GEN_INFO_IMG_PATH,
#            show_shapes=True, show_layer_names=True, dpi=96)
# plot_model(discriminator, to_file=DIS_INFO_IMG_PATH,
#            show_shapes=True, show_layer_names=True, dpi=96)

# make GAN
gan = CTGAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM, num_classes=NUM_CLASS)

# set optimizers and loss functions
gan.compile(
    g_optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    d_optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn=BinaryCrossentropy()
)

# callbacks (implement later)
early_stopping_callback = EarlyStopping(
    patience=20,
    restore_best_weights=True
)
checkpoint_callback = ModelCheckpoint(
    filepath=GAN_WEIGHTS_PATH,
    save_best_only=True,
)

# train model
history = gan.fit(dataset, epochs=EPOCHS, callbacks=[early_stopping_callback, checkpoint_callback])

# save weights for generator and discriminator
discriminator.save(DIS_WEIGHTS_PATH)
generator.save(GEN_WEIGHTS_PATH)

# save performance
with open(GAN_PERF_PATH, 'wb') as f:
    pickle.dump(history.history, f)