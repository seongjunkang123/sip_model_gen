import os
import re

def get_version():
    gen_model_weights_directory = './gen_model_weights'

    try:
        files = os.listdir(gen_model_weights_directory)
    except FileNotFoundError:
        os.makedirs(gen_model_weights_directory)
        files = []

    if not files:
        return 1

    # Assuming filenames are like 'model_v1.pt', 'model_v10.h5', etc.
    # We extract numbers from filenames and find the max.
    max_version = 0
    for file in files:
        # Find all numbers in the filename
        numbers = re.findall(r'\d+', file)
        if numbers:
            # Consider the last number found as the version
            version = int(numbers[-1])
            if version > max_version:
                max_version = version

    v = max_version + 1

    print(f"Version: {v}")
    return v