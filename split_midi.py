import mido
from mido import MidiFile, MidiTrack, MetaMessage, Message
import os

INPUT_FOLDER = "data/jordan_midi"
OUTPUT_FOLDER = "data/jordan_midi_split"

import mido
from mido import MidiFile, MidiTrack, MetaMessage, Message
import os

def split_midi(input_file, segment_duration=5 * 60):
    input_filepath = os.path.join(INPUT_FOLDER, input_file)
    midi = MidiFile(input_filepath)
    
    # Calculate ticks per segment
    ticks_per_beat = midi.ticks_per_beat
    tempo = 500000  # Default tempo (microseconds per beat)
    ticks_per_segment = int((segment_duration * 1000000 / tempo) * ticks_per_beat)
    
    output_files = []
    current_segment = 0
    
    # Calculate total ticks in the file
    total_ticks = max(sum(msg.time for msg in track) for track in midi.tracks)
    
    while current_segment * ticks_per_segment < total_ticks:
        output_midi = MidiFile(type=midi.type, ticks_per_beat=ticks_per_beat)
        segment_start_tick = current_segment * ticks_per_segment
        segment_end_tick = (current_segment + 1) * ticks_per_segment
        
        for track in midi.tracks:
            output_track = MidiTrack()
            output_midi.tracks.append(output_track)
            track_ticks = 0
            last_tick = 0
            
            for msg in track:
                track_ticks += msg.time
                
                if segment_start_tick <= track_ticks < segment_end_tick:
                    new_msg = msg.copy()
                    new_msg.time = track_ticks - max(segment_start_tick, last_tick)
                    output_track.append(new_msg)
                    last_tick = track_ticks
                
                if isinstance(msg, MetaMessage) and msg.type == 'set_tempo':
                    tempo = msg.tempo
                    ticks_per_segment = int((segment_duration * 1000000 / tempo) * ticks_per_beat)
                    segment_end_tick = min((current_segment + 1) * ticks_per_segment, total_ticks)
                    
                    if track_ticks < segment_start_tick:
                        # Add tempo change at the start of the segment if it occurred earlier
                        tempo_msg = MetaMessage('set_tempo', tempo=tempo, time=0)
                        output_track.insert(0, tempo_msg)
            
            # Add end of track message
            if output_track:
                output_track.append(MetaMessage('end_of_track', time=0))
        
        if all(len(track) == 0 for track in output_midi.tracks):
            break

        # Create output folder if it doesn't exist
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        
        output_filename = f"{os.path.splitext(input_file)[0]}_segment_{current_segment + 1}.mid"
        output_midi.save(os.path.join(OUTPUT_FOLDER, output_filename))
        output_files.append(output_filename)
        
        current_segment += 1
    
    return output_files

def process(input_folder, output_folder, segment_length_minutes):
    for file in os.listdir(input_folder):
        if file.endswith('.mid'):
            print(f"Processing {file}...")
            split_midi(file)

# Example usage
if __name__ == '__main__':
    process(INPUT_FOLDER, OUTPUT_FOLDER, segment_length_minutes=5)