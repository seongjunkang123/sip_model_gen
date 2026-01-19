# view generator and discriminator losses respect to epochs.

import pickle, matplotlib.pyplot as plt, os

VERSION = input("Trial Number: ")
HISTORY_PATH = f"./gan_model_performance/gan_v{VERSION}.pkl"

if not os.path.exists(HISTORY_PATH):
    print(f"File not found")
    exit(67)

with open(HISTORY_PATH, 'rb') as f:
    history = pickle.load(f)

# plot history
plt.plot(history['d_loss'], label='Discriminator Loss')
plt.plot(history['g_loss'], label='Generator Loss')
plt.title('GAN Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.grid()

# print min losses
print("min discriminator loss: ", min(history['d_loss']))
print("min generator loss: ", min(history['g_loss']))

HISTORY_IMG_PATH = f"./gan_model_performance/gan_v{VERSION}.png"
if not os.path.exists(HISTORY_IMG_PATH):
    plt.savefig(HISTORY_IMG_PATH)
plt.show()