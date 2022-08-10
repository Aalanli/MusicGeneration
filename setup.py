# %%
import requests
import os
import zipfile

def download_dataset():
    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"

    if not os.path.exists("datasets/maestro"):
        os.makedirs("datasets/maestro")
        response = requests.get(url)
        open("datasets/maestro/data.zip", "wb").write(response.content)
        with zipfile.ZipFile("datasets/maestro/data.zip", 'r') as zip_f:
            zip_f.extractall("datasets/maestro")

import sys
import subprocess
def install_dependencies():
    packages = [
        'wandb',
        'ray',
        'tqdm',
        'py_midicsv'
    ]
    for p in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', p])

