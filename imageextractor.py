# Image Extractor from video file
# By: Ammar Chalifah
# 2020/7/20
import cv2
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--frame_per_pop', default = 24, help = 'Number of frames passed for each image saved.')
parser.add_argument('-i', '--input_path', default = 'sample.mp4', help ='path to video input. Pass 0 for video stream from webcam.')
parser.add_argument('-o', '--output', default = 'cctv1', help = 'output name')
parser.add_argument('-e', '--endframe', default = 7100, help = 'endframe')

args = parser.parse_args()

framecount = 1
frm = int(args.frame_per_pop)

start_time = time.time()
video_capture = cv2.VideoCapture(args.input_path)
while video_capture.isOpened():
    success, frame = video_capture.read()
    if framecount >= int(args.endframe):
        break
    if framecount % frm == 0:
        cv2.imwrite('images/{:s}-{:d}.jpg'.format(args.output, framecount), frame)
    framecount += 1
print("--- %s seconds ---" % (time.time() - start_time))
print('Saved {:d} images'.format(round(framecount/frm)))