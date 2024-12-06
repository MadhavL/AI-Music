import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pickle

STRIDE = 10

def load_midi(seq_name, midi_data_dir):
    filepath = os.path.join(midi_data_dir, f'{"_".join(seq_name.split("_")[:-1])}.pkl')
    with open(filepath, 'rb') as f:
            windows = pickle.load(f)
    print(f"Data loaded from {seq_name}")
    return windows

def load_leapmotion_data(name, leap_data_dir='./data/leap_data/'):
    return np.load(os.path.join(leap_data_dir, f"{name}.npy"))

def normalize_leap(array):
    """
    Normalize a NumPy array from integer range [0, 700] to floating range [-1, 1].
    """
    return (array / 350.0) - 1.0

class MotionAudioDataset(Dataset):
    def __init__(self, seq_names, motion_input_len=100, audio_input_len=200, motion_output_len=20, midi_data_dir='./data/processed_midi_split/', leap_data_dir='./data/leap_data/'):
        self.x_motion = []
        self.x_audio = []
        self.y = []
        for seq_name in seq_names:
            print(f"Loading {seq_name}")
            midi_data = load_midi(seq_name, midi_data_dir = midi_data_dir)[::2] #MIDI is 100fps, audio is 50fps, so for alignment, take every 2nd frame of MIDI
            print(f"Midi windows length: {len(midi_data)}")
            motion_data = normalize_leap(load_leapmotion_data(seq_name, leap_data_dir=leap_data_dir))
            print(f"Motion raw length: {motion_data.shape[0]}")
            min_length = min(motion_data.shape[0] - audio_input_len, len(midi_data))
            for start in range(0, min_length - audio_input_len - 1):
                motion_input = motion_data[start:start + motion_input_len]
                midi_input = midi_data[start]
                if start == 0:
                    print(f"Motion: {motion_input.shape[0]}, Midi: {len(midi_input)}")
                motion_target = motion_data[start + motion_input_len:start + motion_input_len + motion_output_len]
                self.x_motion.append(motion_input)
                self.x_audio.append(midi_input)
                self.y.append(motion_target)
        self.x_motion = torch.tensor(np.array(self.x_motion, dtype=float), dtype=torch.float32)
        self.x_audio = torch.tensor(np.array(self.x_audio, dtype=np.int32), dtype=torch.int32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return self.x_motion.shape[0]

    def __getitem__(self, idx):
        return (self.x_motion[idx], self.x_audio[idx]), self.y[idx]

def create_dataloader(batch_size=48, motion_input_len=100, audio_input_len=200, motion_output_len=20, midi_data_dir='./data/processed_midi_split/', leap_data_dir='./data/leap_data/', shuffle=True):
    seq_names = [os.path.splitext(file)[0] for file in os.listdir(leap_data_dir)]
    print(seq_names)
    dataset = MotionAudioDataset(seq_names, motion_input_len=motion_input_len, audio_input_len=audio_input_len, motion_output_len=motion_output_len, midi_data_dir=midi_data_dir, leap_data_dir=leap_data_dir)
    print(dataset.x_motion.shape)
    print(dataset.x_audio.shape)
    print(dataset.y.shape)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)