from absl import app

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import wandb
import models
import data_utils_midi
from anticipation.convert import midi_to_events
from anticipation.tokenize import tokenize
from hparams import hparams_music_transformer, hparams_motion_transformer

BATCH_SIZE = 1 #128
MIDI_DIR = './data/jordan_midi_split_processed/'
LEAP_DATA_DIR = './data/jordan_midi_leap/'
LEARNING_RATE = 1e-4
EPOCHS = 2500
CHECKPOINT = './ModelsMidi/model'
MODEL_DIR = './ModelsMidi'
USE_CUDA = False
LOAD_MODEL = False

def main(_):
    dataset = data_utils_midi.create_dataloader(batch_size=BATCH_SIZE, motion_input_len=hparams_motion_transformer['motion_input_len'], audio_input_len=hparams_music_transformer['audio_input_len'], motion_output_len=hparams_motion_transformer['supervised_len'], midi_data_dir=MIDI_DIR, leap_data_dir=LEAP_DATA_DIR, shuffle=True)

    model = models.FACTModel_midi(use_cuda=USE_CUDA)
    if USE_CUDA:
        model = nn.DataParallel(model)
        model.to(torch.device("cuda"))

    # Load checkpoint
    if LOAD_MODEL:
        model.load_state_dict(torch.load(CHECKPOINT, map_location=torch.device('cpu')))

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

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
                loss = loss_fn(output[:, :hparams_motion_transformer['supervised_len'], :], Y.cuda()) if USE_CUDA else loss_fn(output[:, :hparams_motion_transformer['supervised_len'], :], Y)
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