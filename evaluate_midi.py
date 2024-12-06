from absl import app
import models
import torch
import data_utils_midi
import numpy as np
import torch.nn as nn
import os

MOTION_INPUT_LEN = 100
CHECKPOINT = './ModelsPerry/model_interrupt'
DATA_DIR = "./data/leap_midi_data"
SEQ = "jd1_task1_train_segment_1_0"
OUTPUT_NAME = 'pt_new_test.npy'

def denormalize_leap(array):
    """
    Denormalize a NumPy array from floating range [-1, 1] to integer range [0, 700].
    """
    return ((array + 1.0) * 350.0).cpu().detach().numpy()

def infer_auto_regressive(model, motion_input, midi, steps=1200, midi_seq_length = 200):
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
      midi_input = torch.tensor(midi[i], dtype=torch.int32).unsqueeze(0)
    #   print(f"midi_input shape: {midi_input}")
      if midi_input.shape[1] < midi_seq_length:
        break
    #   print(f"motion input shape: {motion_input}")
      output = model((motion_input, midi_input))
      output = output[:, 0:1, :]  # only keep the first frame
      outputs.append(output)
      # update motion input
      motion_input = torch.cat([motion_input[:, 1:, :], output], axis=1)
    return torch.cat(outputs, axis=1)

def infer_auto_regressive_interleave(model, motion, midi, steps=1200, midi_seq_length = 200, motion_seq_length = 100):
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
    print(f"motion input shape: {motion_input.shape}")
    for i in range(steps):
      if (i % 2 == 0):
        midi_input = torch.tensor(midi[i], dtype=torch.int32).unsqueeze(0)
        # print(f"Midi input shape: {midi_input.shape}")
        output = model((motion_input, midi_input))
        output = output[:, 0:1, :] # only keep the first frame
      else:
        output = motion[:, i + motion_seq_length, :].unsqueeze(0)
        
      outputs.append(output)
      # update motion input
      motion_input = torch.cat([motion_input[:, 1:, :], output], axis=1)
    
    return torch.cat(outputs, axis=1)

def main(_):
    model = models.FACTModel_midi(use_cuda=False)
    # model = nn.DataParallel(model)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=torch.device('cpu')))
    # model = model.module.to(torch.device("cpu"))
    
    motion_orig = data_utils_midi.load_leapmotion_data(SEQ, leap_data_dir=DATA_DIR)

    # motion = data_utils_midi.load_leapmotion_data(SEQ, leap_data_dir=DATA_DIR)[:100, :] # For fully autoregressive
    motion = data_utils_midi.load_leapmotion_data(SEQ, leap_data_dir=DATA_DIR) # For interleaved
    motion = data_utils_midi.normalize_leap(motion)
    motion = torch.tensor(motion, dtype=torch.float32).unsqueeze(0)

    midi = data_utils_midi.load_midi(SEQ, midi_data_dir="./data/processed_midi_split")
    # midi = torch.tensor(midi, dtype=torch.int32).unsqueeze(0)

    # outputs = infer_auto_regressive(model = model, motion_input = motion, midi = midi, steps = 1200, midi_seq_length = 200)

    outputs = infer_auto_regressive_interleave(model = model, motion = motion, midi = midi, steps = 1200, motion_seq_length = 100, midi_seq_length = 200)

    # denormalize outputs from range [-1, 1] to integers in range [0, 700]
    outputs = denormalize_leap(outputs).squeeze().astype(np.int32)

    np.save(os.path.join("./outputs_new", OUTPUT_NAME), np.concatenate((motion_orig[:120, :], outputs), axis = 0))

if __name__ == '__main__':
  app.run(main)
