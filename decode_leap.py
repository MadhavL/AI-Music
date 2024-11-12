#!/usr/bin/env python3
import numpy as np
import cv2
import simpleaudio as sa
import time
import os

FPS = 50
DIR = "jordan_midi_leap"
DIMENSIONS = 3

def decode_images(images, filename):
    data = []
    print(images.shape)
    images = images.reshape(-1, 48, DIMENSIONS)
    print(images.shape)
    output_image = np.zeros((500, 700, 3), np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID', 'MJPG', etc.
    video = cv2.VideoWriter(f"{filename}.mp4", fourcc, 50, (700, 500))

    for image in images:
        output_image[:, :] = 0
        for point in image:
            cv2.circle(output_image, point[:2], 2, (255, 255, 255), -1)

        video.write(output_image)

    # Release the VideoWriter object
    video.release()

def main():
    for file in os.listdir(DIR):
        print(file)
        filename = file.split(".")[0]
        decode_images(np.load(f"{DIR}/{filename}.npy"), filename)

if __name__ == "__main__":
    main()
