import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

STRIDE = 10

def load_cached_audio_features(seq_name, audio_cache_dir):
    audio_name = seq_name.split("_")[0]
    return np.load(os.path.join(audio_cache_dir, f"{audio_name}.npy")), audio_name

def load_leapmotion_data(name, leap_data_dir='./data/leap_data/'):
    return np.load(os.path.join(leap_data_dir, f"{name}.npy"))

def normalize_leap(array):
    """
    Normalize a NumPy array from integer range [0, 700] to floating range [-1, 1].
    """
    return (array / 350.0) - 1.0

class MotionAudioDataset(Dataset):
    def __init__(self, seq_names, motion_input_len=120, audio_input_len=240, motion_output_len=20, audio_cache_dir='./data/aist_audio_feats/', leap_data_dir='./data/leap_data/'):
        self.x_motion = []
        self.x_audio = []
        self.y = []
        for seq_name in seq_names:
            print(f"Loading {seq_name}")
            motion_data = normalize_leap(load_leapmotion_data(seq_name, leap_data_dir=leap_data_dir))
            audio_data, _ = load_cached_audio_features(seq_name, audio_cache_dir=audio_cache_dir)
            min_length = min(motion_data.shape[0], audio_data.shape[0])
            audio_data = audio_data[-min_length:, :]
            motion_data = motion_data[-min_length:, :]

            seq_len = motion_data.shape[0]
            for start in range(0, seq_len - audio_input_len - 1):
                motion_input = motion_data[start:start + motion_input_len]
                audio_input = audio_data[start:start + audio_input_len]
                motion_target = motion_data[start + motion_input_len:start + motion_input_len + motion_output_len]
                self.x_motion.append(motion_input)
                self.x_audio.append(audio_input)
                self.y.append(motion_target)
        self.x_motion = torch.tensor(np.array(self.x_motion, dtype=float), dtype=torch.float32)
        self.x_audio = torch.tensor(np.array(self.x_audio, dtype=float), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return self.x_motion.shape[0]

    def __getitem__(self, idx):
        return (self.x_motion[idx], self.x_audio[idx]), self.y[idx]

def create_dataloader(batch_size=48, motion_input_len=120, audio_input_len=240, motion_output_len=20, audio_cache_dir='./data/processed_midi/', leap_data_dir='./data/leap_data/', shuffle=True):
    seq_names = [".".join(file.split(".")[:-1]) for file in os.listdir(leap_data_dir)]
    print(seq_names)
    dataset = MotionAudioDataset(seq_names, motion_input_len=motion_input_len, audio_input_len=audio_input_len, motion_output_len=motion_output_len, audio_cache_dir=audio_cache_dir, leap_data_dir=leap_data_dir)
    print(dataset.x_motion.shape)
    print(dataset.x_audio.shape)
    print(dataset.y.shape)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)