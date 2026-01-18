# view generator and discriminator losses over epochs side by side to other trials

import pickle, matplotlib.pyplot as plt, os

VERSION_1 = input("Trial Number: ")
VERSION_2 = input("Trial Number: ")
HISTORY_PATH_1 = f"./gan_model_performance/gan_v{VERSION_1}.pkl"
HISTORY_PATH_2 = f"./gan_model_performance/gan_v{VERSION_2}.pkl"

if not os.path.exists(HISTORY_PATH_1) or not os.path.exists((HISTORY_PATH_2)):
    print(f"File not found")
    exit(67)

with open(HISTORY_PATH_1, 'rb') as f:
    history_1 = pickle.load(f)

with open(HISTORY_PATH_2, 'rb') as f:
    history_2 = pickle.load(f)

# plot histories
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history_1['d_loss'], label='Discriminator Loss')
plt.plot(history_1['g_loss'], label='Generator Loss')
plt.title(f"Trial #{VERSION_1}")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_2['d_loss'], label='Discriminator Loss')
plt.plot(history_2['g_loss'], label='Generator Loss')
plt.title(f"Trial #{VERSION_2}")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# plt.grid()

# print min losses
print(f"Trial #{VERSION_1}")
print("min discriminator loss: ", round(min(history_1['d_loss']), 10))
print("min generator loss: ", round(min(history_1['g_loss'])), 10)

print(f"Trial #{VERSION_2}")
print("min discriminator loss: ", round(min(history_2['d_loss']), 10))
print("min generator loss: ", round(min(history_2['g_loss'])), 10)

# HISTORY_IMG_PATH = f"./gan_model_performance/gan_v{VERSION}.png"
# if not os.path.exists(HISTORY_IMG_PATH):
#     plt.savefig(HISTORY_IMG_PATH)

plt.tight_layout()
plt.show()