from absl import app
import models
import torch
import data_utils
import numpy as np
import torch.nn as nn
import os

MOTION_INPUT_LEN = 120
AUDIO_INPUT_LEN = 240
CHECKPOINT = './ModelsJul16/model_105_0.0000264'
AUDIO_CACHE_DIR = './data/aist_audio_feats/'

DATA_DIR = "./data/leap_data_madhav_unseen"
SEQ = "06 DARE (Radio Edit)_0"
OUTPUT_NAME = 'pt_new_madhav_105e_ni_dare.npy'

def denormalize_leap(array):
    """
    Denormalize a NumPy array from floating range [-1, 1] to integer range [0, 700].
    """
    return ((array + 1.0) * 350.0).cpu().detach().numpy()

def infer_auto_regressive(model, motion_input, audio, steps=1200, audio_seq_length = 240):
    """Predict sequences from inputs in an auto-regressive manner. 

    This function should be used only during inference. During each forward step, 
    only the first frame was kept. Inputs are shifted by 1 frame after each forward.

    Returns:
      Final output after the auto-regressive inference. A tensor with shape 
      [batch_size, steps, motion_feature_dimension]
      will be return.
    """
    outputs = []
    for i in range(steps):
      audio_input = audio[:, i: i + audio_seq_length]
      if audio_input.shape[1] < audio_seq_length:
        break
      
      output = model((motion_input, audio_input))
      output = output[:, 0:1, :]  # only keep the first frame
      outputs.append(output)
      # update motion input
      motion_input = torch.cat([motion_input[:, 1:, :], output], axis=1)
    return torch.cat(outputs, axis=1)

def infer_auto_regressive_interleave(model, motion, audio, steps=1200, audio_seq_length = 240, motion_seq_length = 120):
    """Predict sequences from inputs in an auto-regressive manner. 

    This function should be used only during inference. During each forward step, 
    only the first frame was kept. Inputs are shifted by 1 frame after each forward.

    Returns:
      Final output after the auto-regressive inference. A tensor with shape 
      [batch_size, steps, motion_feature_dimension]
      will be return.
    """
    outputs = []
    motion_input = motion[:, :motion_seq_length]
    for i in range(steps):
      if (i % 2 == 0):
        audio_input = audio[:, i: i + audio_seq_length]
        if audio_input.shape[1] < audio_seq_length:
            break
        output = model((motion_input, audio_input))
        output = output[:, 0:1, :] # only keep the first frame
      else:
        output = motion[:, i + motion_seq_length, :].unsqueeze(0)
        
      outputs.append(output)
      # update motion input
      motion_input = torch.cat([motion_input[:, 1:, :], output], axis=1)
    
    return torch.cat(outputs, axis=1)

def main(_):
    # model = models.fact_model_custom(motion_input_len=MOTION_INPUT_LEN, audio_input_len=AUDIO_INPUT_LEN, use_cuda=False)
    model = models.FACTModel(use_cuda=False)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=torch.device('cpu')))
    model = model.module.to(torch.device("cpu"))
    
    motion_orig = data_utils.load_leapmotion_data(SEQ, leap_data_dir=DATA_DIR)

    motion = data_utils.load_leapmotion_data(SEQ, leap_data_dir=DATA_DIR)[:120, :]
    # motion = data_utils.load_leapmotion_data(SEQ, leap_data_dir=DATA_DIR)
    motion = data_utils.normalize_leap(motion)
    motion = torch.tensor(motion, dtype=torch.float32).unsqueeze(0)

    audio, _ = data_utils.load_cached_audio_features(SEQ, audio_cache_dir=AUDIO_CACHE_DIR)
    audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

    outputs = infer_auto_regressive(model = model, motion_input = motion, audio = audio, steps = 1200, audio_seq_length = 240)

    # outputs = infer_auto_regressive_interleave(model = model, motion = motion, audio = audio, steps = 1200, motion_seq_length = 120, audio_seq_length = 240)

    # denormalize outputs from range [-1, 1] to integers in range [0, 700]
    outputs = denormalize_leap(outputs).squeeze().astype(np.int32)

    np.save(os.path.join("./outputs_new", OUTPUT_NAME), np.concatenate((motion_orig[:120, :], outputs), axis = 0))

if __name__ == '__main__':
  app.run(main)
