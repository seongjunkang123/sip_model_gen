import os

def get_version():
    gen_model_weights_directory = './gen_model_weights'

    try: 
        sorted_files = sorted(os.listdir(gen_model_weights_directory))
    except FileNotFoundError:
        os.makedirs(gen_model_weights_directory)
        sorted_files = []
    
    last_index = len(sorted_files) - 1
    if last_index < 0:
        v = 1
    else:
        file = sorted_files[last_index]
        v = int(file[5]) + 1

    return v