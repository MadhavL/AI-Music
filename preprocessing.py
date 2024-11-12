from absl import app
from absl import flags
from absl import logging

import os
import random
import numpy as np
import librosa

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'audio_dir', './data/music/', 
    'Path to the AIST wav files.')
flags.DEFINE_string(
    'audio_cache_dir', './data/aist_audio_feats/', 
    'Path to cache dictionary for audio features.')

RNG = np.random.RandomState(42)

def cache_audio_features(seq_names, tempos):
    # Extract features from the audio data using librosa & write it to the aist_audio_feats folder (default location)
    FPS = 60
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    EPS = 1e-6

    def _get_tempo(audio_name):
        """Get tempo (BPM) for a music by parsing music name."""
        assert len(audio_name) == 4
        if audio_name[0:3] in ['mBR', 'mPO', 'mLO', 'mMH', 'mLH', 'mWA', 'mKR', 'mJS', 'mJB']:
            return int(audio_name[3]) * 10 + 80
        elif audio_name[0:3] == 'mHO':
            return int(audio_name[3]) * 5 + 110
        else: assert False, audio_name

    for audio_name in seq_names:
        print(f"Processing {audio_name}")
        save_path = os.path.join(FLAGS.audio_cache_dir, f"{audio_name}.npy")
        if os.path.exists(save_path):
            continue
        data, _ = librosa.load(os.path.join(FLAGS.audio_dir, f"{audio_name}.wav"), sr=SR)
        envelope = librosa.onset.onset_strength(data, sr=SR)  # (seq_len,)
        mfcc = librosa.feature.mfcc(data, sr=SR, n_mfcc=20).T  # (seq_len, 20)
        chroma = librosa.feature.chroma_cens(
            data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T  # (seq_len, 12)

        peak_idxs = librosa.onset.onset_detect(
            onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
        peak_onehot = np.zeros_like(envelope, dtype=np.float32)
        peak_onehot[peak_idxs] = 1.0  # (seq_len,)

        tempo, beat_idxs = librosa.beat.beat_track(
            onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH,
            start_bpm=tempos[audio_name], tightness=100)
        
        # start_bpm=_get_tempo(audio_name)
        beat_onehot = np.zeros_like(envelope, dtype=np.float32)
        beat_onehot[beat_idxs] = 1.0  # (seq_len,)

        audio_feature = np.concatenate([
            envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]
        ], axis=-1)
        np.save(save_path, audio_feature)


def main(_):
    # create list
    seq_names = [".".join(file.split(".")[:-1]) for file in os.listdir(FLAGS.audio_dir)]

    # tempos
    tempos = None
    with open('./data/tempos', 'r') as file:
        tempos = {line.split(': ')[0]: int(line.split(': ')[1]) for line in file}
    
    # create audio features
    print ("Pre-compute audio features ...")
    os.makedirs(FLAGS.audio_cache_dir, exist_ok=True)
    cache_audio_features(seq_names, tempos)
        
if __name__ == '__main__':
  app.run(main)