from pathlib import Path

def get_config():
    return{
        "batch_size": 4,
        "num_epoch": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weight",
        "model_filename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch:int):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = config['model_filename']
    return str(Path('.') / model_folder / model_filename)