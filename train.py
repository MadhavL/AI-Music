from absl import app

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import wandb
import models
import data_utils
from anticipation.convert import midi_to_events
from anticipation.tokenize import tokenize


MOTION_INPUT_LEN = 120
AUDIO_INPUT_LEN = 240
MOTION_OUTPUT_LEN = 20
BATCH_SIZE = 1 #128
AUDIO_CACHE_DIR = './data/aist_audio_feats/'
LEAP_DATA_DIR = './data/leap_data_perry/'
LEARNING_RATE = 1e-4
EPOCHS = 2500
CHECKPOINT = './ModelsJul16/model'
MODEL_DIR = './ModelsPerry'

def main(_):
    
    dataset = data_utils.create_dataloader(batch_size=BATCH_SIZE, motion_input_len=MOTION_INPUT_LEN, motion_output_len=MOTION_OUTPUT_LEN, audio_cache_dir=AUDIO_CACHE_DIR, leap_data_dir=LEAP_DATA_DIR, shuffle=True)

    events = midi_to_events('examples/test2.mid')
    print(f"Events:\n{len(events)}\n{events}")

    # model = models.fact_model_custom(motion_input_len=MOTION_INPUT_LEN, audio_input_len=AUDIO_INPUT_LEN, use_cuda=True)
    model = models.FACTModel(use_cuda=True)
    model = nn.DataParallel(model)
    model.to(torch.device("cuda"))

    print(model)

    # Load checkpoint
    # model.load_state_dict(torch.load(CHECKPOINT, map_location=torch.device('cpu')))

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    #Implement learning rate schedule

    wandb.init(
        # set the wandb project where this run will be logged
        project="ai_music_motion",

        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "FACT",
        "epochs": EPOCHS,
        }
    )

    try:
        loss = None
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}")

            # Save model checkpoints
            if epoch != 0 and epoch % 5 == 0:
                print("Saving model...")
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"model_{epoch}_{loss.item():.7f}"))

            # Implement crude learning rate schedule
            if epoch > 50 and loss.item() < 0.0005 and LEARNING_RATE == 1e-4:
                LEARNING_RATE == 1e-5
                for g in optimizer.param_groups:
                    g['lr'] = 1e-5

            # Train
            for batch, (X, Y) in enumerate(dataset):
                output = model(X)
                loss = loss_fn(output[:, :MOTION_OUTPUT_LEN, :], Y.cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss.item())
                wandb.log({"loss": loss.item()})
                
        print("Saving model...")
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"model_2500_{loss.item():.7f}"))

    except KeyboardInterrupt:
        print("Saving model...")
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'model_interrupt'))
        sys.exit() 

if __name__ == '__main__':
  app.run(main)