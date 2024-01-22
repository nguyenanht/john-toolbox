from pathlib import Path

DATA_FOLDER = "/work/data"


def get_config():
    return {
        "batch_size": 8,  # Size of each batch for training
        "num_epochs": 2,  # Number of training epochs
        "lr": 10**-4,  # Learning rate for the optimizer
        "seq_len": 400,  # Maximum sequence length for the model
        "d_model": 512,  # Dimensionality of the token embeddings
        "datasource": "opus_books",  # Source of the training data
        "lang_src": "en",  # Source language code (e.g., English)
        "lang_tgt": "fr",  # Target language code (e.g., French)
        "model_folder": "weights",  # Directory to store model weights
        "model_basename": "tmodel_",  # Base name for the saved model files
        "preload": None,  # Preloaded model weights if available, otherwise None
        "tokenizer_file": "tokenizer_{0}.json",  # Filename pattern for saving the tokenizer
        "experiment_name": "runs/tmodel",  # Name for the experiment, used for logging
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{DATA_FOLDER}/{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return f"{model_folder}/{model_filename}"


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{DATA_FOLDER}/{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
