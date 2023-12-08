from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_{0:02d}.pt",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
    }

def get_weights_file_path(config, epoch: str) -> str:
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = model_basename.format(epoch)
    return str(Path('.') / model_folder / model_filename)

def get_latest_weights_file_path(config) -> str:
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    # Check all files in the model folder
    model_files = Path(model_folder).glob(f"*.pt")
    # Sort by epoch number (ascending order)
    model_files = sorted(model_files, key=lambda x: int(x.stem.split('_')[-1]))
    if len(model_files) == 0:
        return None
    # Get the last one
    model_filename = model_files[-1]
    return str(model_filename)