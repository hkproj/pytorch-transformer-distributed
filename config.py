from pathlib import Path
from dataclasses import dataclass

@dataclass
class ModelConfig:

    batch_size: int # Batch size
    num_epochs: int # Number of epochs to train
    lr: float # Learning rate
    seq_len: int # Sequence length
    d_model: int # Size of the embedding vector
    lang_src: str # Source language
    lang_tgt: str # Target language
    model_folder: str # Folder where to save the checkpoints
    model_basename: str # Basename of the checkpoint files
    preload: str # Preload weights from a previous checkpoint
    tokenizer_file: str # Path where to save the tokenizer
    local_rank: int = -1 # LOCAL_RANK assigned by torchrun
    global_rank: int = -1 # RANK assigned by torchrun

def get_default_config() -> ModelConfig:

    return ModelConfig(
        batch_size=4,
        num_epochs=30,
        lr=10**-4,
        seq_len=350,
        d_model=512,
        lang_src="en",
        lang_tgt="it",
        model_folder="weights",
        model_basename="tmodel_{0:02d}.pt",
        preload="latest",
        tokenizer_file="tokenizer_{0}.json",
    )

def get_weights_file_path(config: ModelConfig, epoch: str) -> str:
    model_folder = config.model_folder
    model_basename = config.model_basename
    model_filename = model_basename.format(epoch)
    return str(Path('.') / model_folder / model_filename)

def get_latest_weights_file_path(config: ModelConfig) -> str:
    model_folder = config.model_folder
    model_basename = config.model_basename
    # Check all files in the model folder
    model_files = Path(model_folder).glob(f"*.pt")
    # Sort by epoch number (ascending order)
    model_files = sorted(model_files, key=lambda x: int(x.stem.split('_')[-1]))
    if len(model_files) == 0:
        return None
    # Get the last one
    model_filename = model_files[-1]
    return str(model_filename)