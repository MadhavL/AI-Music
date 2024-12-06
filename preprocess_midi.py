import os
from anticipation.convert import midi_to_events
from anticipation.tokenize import tokenize
from anticipation.config import *
from anticipation.vocab import *
import numpy as np
import timeit
import pickle

FOLDER = "data/jordan_midi_split"
OUT_FOLDER = "data/jordan_midi_split_processed"
TIME_RESOLUTION = 100  #MIDI ticks corresponding to 1 second
SECONDS = 4  # MIDI sequence length in seconds
STRIDE = 0.01  # Stride length in seconds
CONTEXT_LENGTH = 204 # Number of tokens in a window
# 1 bar = 2s = 200 ticks @ 120bpm. 1 beat = 0.5s = 50 ticks
# Number of bars covered by the windows = (Number of bars - 2)
# Stride in bars = Stride in seconds / 2
# Number of windows = [(Num bars - 2) * (1 / Stride in bars) + 1] 
# OR 
# [(Num bars - 2) * (2 / Stride in seconds) + 1]
# Num bars = ([Num windows - 1] * [Stride in seconds / 2]) + 2

def extract_windows(filepath):
    # Convert MIDI to events for entire file
    events = midi_to_events(filepath)
    # print(f"Raw: {events}\n")
    print(f"Number of events: {len(events)}")

    # Initialize start and end of window
    end = TIME_RESOLUTION * SECONDS
    start = 0
    
    # Start_index tracks the index to start the next window at so we don't have to start from the beginning of the events list every loop
    start_index = 0
    set_start_index = True

    # Initialize list of windows
    windows = [] # List of lists of events
    window = [] # List of events

    # Find the end of the last event of the events list (rounded to nearest 100 (i.e. time_resolution)). Hardcoded the 100 for now.
    last_event_timestamp = round(max(events[i] + (events[i+1] - DUR_OFFSET) for i in range(0, len(events), 3)), -2) # Finding the event which when time + duration taken into account is the latest
    # print(f"End of last event: {last_event_timestamp}"")

    # Keep going until the end of the last event
    while end <= last_event_timestamp:
        # Iterate through the indices of timestamps of the events
        # print(f"Start: {start}, End: {end}, start_index: {start_index}")
        for i in range(start_index, len(events), 3):
            # Find the first event that will be in the next window and set the start index to that index
            if set_start_index and events[i] > start + int(STRIDE * TIME_RESOLUTION):
                start_index = i
                set_start_index = False
                # print(f"Start index set to {start_index}")
            
            # If the event is within the window, add it to the current window
            if events[i] >= start and events[i] < end:
                window.extend(events[i:i+3])
                # If the event is the last event in the events list, end the window
                if i == len(events) - 3:
                    # Go to next window
                    end += int(STRIDE * TIME_RESOLUTION)
                    start += int(STRIDE * TIME_RESOLUTION)
                    # print(f"Window ended. Unprocessed: {window}\n")

                    # Post-process the window (pad / truncate to context_length)
                    # print(f"Unprocessed length: {len(window)}")
                    processed_window = pad_or_truncate(window)
                    # print(f"Processed: {processed_window}\n")
                    # print(f"Processed length: {len(processed_window)}")

                    # Add the processed window to the list of windows
                    windows.append(processed_window)
                    window = []
                    set_start_index = True
                    break
            
            # If not, since we are going in order, we can end the window, add it to the list, and break the loop
            else:
                # Go to next window
                end += int(STRIDE * TIME_RESOLUTION)
                start += int(STRIDE * TIME_RESOLUTION)
                # print(f"Window ended. Unprocessed: {window}\n")

                # Post-process the window (pad / truncate to context_length)
                # print(f"Unprocessed length: {len(window)}")
                processed_window = pad_or_truncate(window)
                # print(f"Processed: {processed_window}\n")
                # print(f"Processed length: {len(processed_window)}")

                # Add the processed window to the list of windows
                windows.append(processed_window)
                window = []
                set_start_index = True
                break
            
    return windows

# Take a 4s Midi window as input
# Pad or truncate to context_length
# Relativize time
def pad_or_truncate(window):
    while len(window) > CONTEXT_LENGTH:
        del window[0:3]

    try:
        start_time = min(window[0::3])
        for index in range(0, len(window), 3):
            window[index] -= start_time
    except:
        pass
    
    while len(window) < CONTEXT_LENGTH:
        window.extend([TIME_RESOLUTION * SECONDS, DUR_OFFSET+0, REST])
    
    return window

# Save the window
def save_windows(windows, filename):
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)
    filepath = os.path.join(OUT_FOLDER, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(windows, f)
    print(f"Data saved to {filename}")

#Load windows
def load_windows(filename):
    filepath = os.path.join(OUT_FOLDER, filename)
    with open(filepath, 'rb') as f:
        windows = pickle.load(f)
    print(f"Data loaded from {filename}")
    return windows

def process(folder):
    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        if filepath.endswith('.mid'):
            print(f"Processing {file}...")
            windows = extract_windows(filepath)
            # print(windows)
            print(f"Number of windows: {len(windows)}")

            save_windows(windows, f"{os.path.splitext(file)[0]}.pkl")

if __name__ == "__main__":
    process(FOLDER)
