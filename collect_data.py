#!/usr/bin/env python3
import sys
import leap
import numpy as np
import cv2
import os
import time
import random
import pygame
import argparse

# Value to pad the data for each joint that can't be found in the frame
PAD_VALUE = (0, 0, 0)
FPS = 50

_TRACKING_MODES = {
    leap.TrackingMode.Desktop: "Desktop",
    leap.TrackingMode.HMD: "HMD",
    leap.TrackingMode.ScreenTop: "ScreenTop",
}

class Canvas:
    def __init__(self, input, random_order):
        # For image rendering
        self.name = "Music Motion Data Collector"
        self.screen_size = [500, 700]
        self.hands_colour = (255, 255, 255)
        self.font_colour = (0, 255, 44)
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)
        self.tracking_mode = None
        # For data collection
        self.data = []
        self.current_song = {"name": "", "data": [], "total_time": 0}
        self.songs = os.listdir(input)
        self.music_dir = input
        self.random_order = random_order
        self.playing = False

    def set_tracking_mode(self, tracking_mode):
        self.tracking_mode = tracking_mode

    def get_joint_position(self, bone):
        if bone:
            return int(bone.x + (self.screen_size[1] / 2)), int(bone.z + (self.screen_size[0] / 2)), int(bone.y)
        else:
            return None
        
    def check_audio(self):
        pygame_busy = pygame.mixer.music.get_busy()
        if not self.playing and pygame_busy:
            # self.current_song['total_time'] = time.time() # Disabled after getting the time from the audio
            self.playing = True
        
        elif not pygame_busy and self.playing:
            self.audio_finished()

    # For moving average alignment of audio & motion data (not needed for now)
    def align_time(self):
        curr_time = time.time()
        if curr_time - self.current_song['total_time'] > 1:
            print(self.current_song['data'])
            self.current_song.update({"alignment": (len(self.current_song['data']), curr_time - self.current_song['total_time'])})

    def render_hands(self, event):
        self.check_audio()

        # if self.playing and "alignment" not in self.current_song:
        #     self.align_time()

        # Clear the previous image
        self.output_image[:, :] = 0

        cv2.putText(
            self.output_image,
            f"Tracking Mode: {_TRACKING_MODES[self.tracking_mode]}",
            (10, self.screen_size[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.font_colour,
            1,
        )

        if len(event.hands) == 0 and self.playing:
            # Add all 0s to the data if no hands are detected
            self.current_song['data'].append([PAD_VALUE] * 48)
            return
        
        points = []

        # Left hand first
        hands = event.hands
        left_hand = None
        right_hand = None
        for hand in hands:
            if hand.type == leap.HandType.Left:
                left_hand = hand
            if hand.type == leap.HandType.Right:
                right_hand = hand
        
        if left_hand:
            left_hand_frame = self.process_hand(left_hand)
            if self.playing:
                points.extend(left_hand_frame) # Add the left hand to the data
            
        elif self.playing:
            points.extend([PAD_VALUE] * 24) # Add all 0s to the data if no left hand is detected

        if right_hand:
            right_hand_frame = self.process_hand(right_hand)
            if self.playing:
                points.extend(right_hand_frame)

        elif self.playing:
            points.extend([PAD_VALUE] * 24)
        
        # Add the frame to our data if a song is playing
        if self.playing:
            self.current_song['data'].append(points)

    def process_hand(self, hand):
        result = []
        for index_digit in range(0, 5):
                digit = hand.digits[index_digit]
                for index_bone in range(0, 4):
                    bone = digit.bones[index_bone]
            
                    prev_joint = self.get_joint_position(bone.prev_joint)
                    next_joint = self.get_joint_position(bone.next_joint)
                    if next_joint:
                        cv2.circle(self.output_image, next_joint[:-1], 2, self.hands_colour, -1)
                        result.append(next_joint)
                    else:
                        result.append(PAD_VALUE) # Add a dummy point if we don't have a previous joint

                    if prev_joint:
                        cv2.circle(self.output_image, prev_joint[:-1], 2, self.hands_colour, -1)

                    # Add the previous joint to the points list, only for the first bone of each digit (and not for the thumb since it is a duplicate)
                    if index_bone == 0 and index_digit != 0:
                        if prev_joint:
                            result.append(prev_joint)
                        else:
                            result.append(PAD_VALUE) # Add a dummy point if we don't have a previous joint
        return result

    def audio_finished(self):
        print("Finished Song! Press 'n' for next")
        # self.current_song['total_time'] = time.time() - self.current_song['total_time'] # Disabled after getting the time from the audio
        self.data.append(self.current_song)
        self.current_song = {"name": "", "data": [], "total_time": 0}
        self.playing = False
        
    def play_audio(self):
        if self.random_order:
            song = random.choice(self.songs)
        else:
            song = self.songs.pop(0)
            self.songs.append(song)
        print(f"Playing {song}")
        self.current_song['name'] = song.split(".")[0]
        
        # Load the song into a sound object
        songObject = pygame.mixer.Sound(os.path.join(self.music_dir, song))
        # Get the length of the song
        length = songObject.get_length()
        self.current_song['total_time'] = length
        # Delete the sound object
        del songObject

        # Load the song into the mixer & play it
        pygame.mixer.music.load(os.path.join(self.music_dir, song))
        pygame.mixer.music.play()

class TrackingListener(leap.Listener):
    def __init__(self, canvas):
        self.canvas = canvas

    def on_connection_event(self, event):
        pass

    def on_tracking_mode_event(self, event):
        self.canvas.set_tracking_mode(event.current_tracking_mode)
        print(f"Tracking mode changed to {_TRACKING_MODES[event.current_tracking_mode]}")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()

        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        self.canvas.render_hands(event)

def interpolate_fps(data, x_fps, y_fps):
    factor = x_fps / y_fps
    new_length = int(len(data) / factor)
    indices = np.linspace(0, len(data) - 1, new_length).astype(int)
    return data[indices]

# Export the data
def export_data(all_songs, output_dir):
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)  
    
    for i, song in enumerate(all_songs):
        # First, find the average fps of the song and downsample to defined FPS
        print(f"{len(song['data'])} frames in {song['name']}")
        time = song['total_time']
        print(f"Total time: {time}")
        avg_fps = len(song['data']) / time
        print(f"Average FPS: {avg_fps}")

        data = np.array(song['data'])
        data = interpolate_fps(data, avg_fps, FPS)

        # Now, reshape the data and save it to disk
        name = song['name']
        # JUST FOR SEED MOTION:
        data = data[100:200, :, :]
        data = data.reshape(-1, data.shape[1] * 3)
        print(data.shape)
        num_files = sum(name in filename for filename in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, filename)))
        np.save(os.path.join(output_dir, f"{name}_{num_files}"), data)

def main():
    parser = argparse.ArgumentParser(description="Collect data for leap motion")

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the input data folder'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to the output data folder'
    )

    parser.add_argument(
        '--random_order',
        action='store_true',
        help='Whether to play songs randomly or in order'
    )

    args = parser.parse_args()

    # Initialize pygame
    pygame.init()
    pygame.mixer.init()

    output_dir = "leap_data"
    canvas = Canvas(input=args.input, random_order=args.random_order)

    print(canvas.name)
    print("")
    print("Press <key> in visualiser window to:")
    print("  x: Exit")
    print("  h: Select HMD tracking mode")
    print("  s: Select ScreenTop tracking mode")
    print("  d: Select Desktop tracking mode")
    print("  f: Toggle hands format between Skeleton/Dots")

    tracking_listener = TrackingListener(canvas)

    connection = leap.Connection()
    connection.add_listener(tracking_listener)

    running = True

    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)
        canvas.set_tracking_mode(leap.TrackingMode.Desktop)
        try:
            while running:
                cv2.imshow(canvas.name, canvas.output_image)
                key = cv2.waitKey(1)
                if key == ord("x"):
                    export_data(canvas.data, output_dir=args.output)
                    break
                if key == ord("n"):
                    canvas.play_audio()
                elif key == ord("h"):
                    connection.set_tracking_mode(leap.TrackingMode.HMD)
                elif key == ord("s"):
                    connection.set_tracking_mode(leap.TrackingMode.ScreenTop)
                elif key == ord("d"):
                    connection.set_tracking_mode(leap.TrackingMode.Desktop)
                elif key == ord("f"):
                    canvas.toggle_hands_format()
        
        except KeyboardInterrupt:
            export_data(canvas.data, output_dir, output_dir=args.output)
            sys.exit(0)

if __name__ == "__main__":
    main()
