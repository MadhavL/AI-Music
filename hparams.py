from torch import torch, device as d
from anticipation import vocab

# get device
if torch.backends.cuda.is_built():
    dev = "cuda"
else:
    dev = "cpu"
device = d(dev)

# default hparams for the music transformer model
hparams_music_transformer = {
    "d_model": 128,
    "num_layers": 3,
    "num_heads": 8,
    "d_ff": 512,
    "max_rel_dist": 1024,
    "max_abs_position": 0,
    "vocab_size": vocab.VOCAB_SIZE,
    "bias": True,
    "dropout": 0.1,
    "layernorm_eps": 1e-6,
    "audio_input_len": 200
}

# default hparams for motion transformer
hparams_motion_transformer = {
    "d_model": 800,
    "num_layers": 2,
    "num_heads": 10,
    "intermediate_size": 800,
    "initializer_range": 0.02,
    "motion_input_len" : 100,
    "motion_num_features": 144,
    "supervised_len" : 20
}

hparams_cross_modal_transformer = {
    "d_model": 800,
    "num_layers": 12,
    "num_heads": 10,
    "intermediate_size": 800,
    "initializer_range": 0.02,
}



# hparams for music transformer model - significantly larger
hparams_large = {
    "d_model": 256,
    "num_layers": 6,
    "num_heads": 8,
    "d_ff": 1024,
    "max_rel_dist": 1024,
    "max_abs_position": 0,
    "vocab_size": vocab.VOCAB_SIZE,
    "bias": True,
    "dropout": 0.1,
    "layernorm_eps": 1e-6
}