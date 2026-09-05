import pickle

import numpy as np
import pandas as pd
import os

from utils import get_version

def add_latest_to_spreadsheet(version: int):
    output_path = './results_summary.csv'
    path = f"./gan_model_performance/gan_v{version}.pkl"

    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with open(path, 'rb') as f:
        history = pickle.load(f)

    min_g_loss = min(history['g_loss'])
    min_d_loss = min(history['d_loss'])
    avg_g_loss = np.average(history['g_loss'])
    avg_d_loss = np.average(history['d_loss'])

    new_row = pd.DataFrame([{
        'version': f'v{version}',
        'min_g_loss': min_g_loss,
        'min_d_loss': min_d_loss,
        'avg_g_loss': avg_g_loss,
        'avg_d_loss': avg_d_loss,
    }])

    new_row.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)

def convert_to_spreadsheet():
    version_start = 1
    version_end = get_version() - 1

    results = []

    for i in range(version_start, version_end + 1):
        path = f"./gan_model_performance/gan_v{i}.pkl"
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        with open(path, 'rb') as f:
            history = pickle.load(f)

        min_g_loss = min(history['g_loss'])
        min_d_loss = min(history['d_loss'])

        avg_g_loss = np.average(history['g_loss'])
        avg_d_loss = np.average(history['d_loss'])

        results.append({
            'version': f'v{i}',
            'min_g_loss': min_g_loss,
            'min_d_loss': min_d_loss,
            'avg_g_loss': avg_g_loss,
            'avg_d_loss': avg_d_loss,
        })

    df = pd.DataFrame(results)

    output_path = './results_summary.csv'
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    convert_to_spreadsheet()