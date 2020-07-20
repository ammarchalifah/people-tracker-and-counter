#Plot object detection bounding boxes on a photo
#2020/07/2020
#by Ammar Chalifah

import matplotlib.pyplot as plt
import argparse
from detection_api import ssd_net
import cv2
import time

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', default = 'ssd_inception_v2_coco_2017_11_17', help = 'Model name.')
parser.add_argument('-i', '--input_path', default = 'sample.jpg', help ='path of file')

args = parser.parse_args()

image = cv2.imread(args.input_path)

start_time = time.time()
new_image = ssd_net(image)
print('------ {:f} seconds ------'.format(time.time() - start_time))

while True:
    cv2.imshow('object detection', new_image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
