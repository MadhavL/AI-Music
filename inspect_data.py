import os
from anticipation.convert import midi_to_events
from anticipation.tokenize import tokenize
from tqdm import tqdm

import os
import numpy as np
import mido
from collections import defaultdict
import matplotlib.pyplot as plt

FOLDER = "data/jordan_data"
TIME_RESOLUTION = 100  # You can adjust this if needed - check if correct
SECONDS = 2  # Sequence length in seconds
STRIDE = 1  # Stride length in seconds

def process_midi_file(filepath):
    # Convert MIDI to events
    events = midi_to_events(filepath)
    # print(events)
    
    # Calculate the number of ticks corresponding to the sequence length
    ticks_per_second = TIME_RESOLUTION
    sequence_ticks = SECONDS * ticks_per_second
    stride_ticks = STRIDE * ticks_per_second
    
    # List to store the number of tokens per sequence
    token_counts = []
    
    # Iterate through the events in sequences of 2-second windows
    for start in range(0, max(events[0::3]), stride_ticks):
        end = start + sequence_ticks
        # Count the number of tokens within the current 2-second window
        count = sum(start <= event < end for event in events[0::3])
        token_counts.append(count*3) # Multiplying by 3 here for tokens length rather than num events
        # print(f"start: {start}, end: {end}, count: {count}")
    
    return token_counts

def analyze_folder(folder):
    all_token_counts = []

    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        if filepath.endswith('.mid'):
            print(f"Processing {file}...")
            token_counts = process_midi_file(filepath)
            all_token_counts.extend(token_counts)
    
    # Convert to a NumPy array for easy statistical analysis
    token_counts = np.array(all_token_counts)
    
    # Calculate basic statistics
    min_tokens = np.min(token_counts)
    max_tokens = np.max(token_counts)
    mean_tokens = np.mean(token_counts)
    std_tokens = np.std(token_counts)

    # Print statistics
    print(f"Min tokens in a 2-second sequence: {min_tokens}")
    print(f"Max tokens in a 2-second sequence: {max_tokens}")
    print(f"Mean tokens in a 2-second sequence: {mean_tokens:.2f}")
    print(f"Std tokens in a 2-second sequence: {std_tokens:.2f}")

    # Plot the distribution
    plt.hist(token_counts, bins=30, alpha=0.7, color='blue')
    plt.title('Distribution of Token Counts in 2-Second Sequences')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    # print(midi_to_events('examples/test3.mid'))
    analyze_folder(FOLDER)