"""Setup important environmental variables"""
import os

import torch
from dotenv import find_dotenv, load_dotenv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load environmental variables from a .env file
load_dotenv(find_dotenv())

# declare global variables
ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))
DATASETS_FOLDER = os.environ.get("DATASETS_FOLDER", f"{ROOT_FOLDER}/datasets")
TENSORS_FOLDER = os.environ.get("TENSORS_FOLDER", f"{ROOT_FOLDER}/features")
RESULTS_FOLDER = os.environ.get("RESULTS_FOLDER", f"{ROOT_FOLDER}/results")
MODELS_FOLDER = os.environ.get("MODELS_FOLDER", f"{ROOT_FOLDER}/models")
IMAGES_FOLDER = os.environ.get("IMAGES_FOLDER", f"{ROOT_FOLDER}/images")

# the .env file should look something like this:
# DATASETS_FOLDER="/path/to/the/datasets/folder"
if __name__ == "__main__":
    print(DATASETS_FOLDER)
    print(ROOT_FOLDER)