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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history['d_loss'], label='Discriminator Loss')
ax1.plot(history['g_loss'], label='Generator Loss')
ax1.set_title('WGAN-GP Loss Over Epochs')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

if 'gp' in history:
    ax2.plot(history['gp'], label='Gradient Penalty', color='green')
    ax2.set_title('Gradient Penalty Over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('GP')
    ax2.legend()

plt.tight_layout()
# plt.grid()

# print min losses
print("min discriminator loss: ", min(history['d_loss']))
print("min generator loss: ", min(history['g_loss']))

HISTORY_IMG_PATH = f"./gan_model_performance/gan_v{VERSION}.png"
if not os.path.exists(HISTORY_IMG_PATH):
    plt.savefig(HISTORY_IMG_PATH)
plt.show()