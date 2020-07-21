import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import time
import argparse
import json
import pandas as pd

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from functions import label_map_util
from functions import visualization_utils as vis_util
from functions.tracker import box_to_centoroid

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', default = 'ssd mobilenet', help = 'Model name to be used. Choose between [ssd inception, ssd mobilenet, faster rcnn resnet]')
parser.add_argument('-i', '--input_path', default = 'videos/lobbyselatan.avi', help ='path of file')
parser.add_argument('-s', '--save_bool', default = False, help = 'log detection data to a csv file.')

args = parser.parse_args()

#------------VIDEO STREAM--------------
# Define the video stream
cap = cv2.VideoCapture(args.input_path)  # Change only if you have more than one webcams
#Target size
target_w = 800
target_h = 600

#Initialize empty dataframe to record results
#------------LOGGING VARIABLE----------------
res_df = pd.DataFrame(columns=['boxes', 'classes','scores','num_detections'])

#Model choosing
#------------DETECTION MODEL-----------------
modelname = {
    'ssd mobilenet':'ssd_mobilenet_v1_ppn_2018_07_03',
    'ssd inception':'ssd_inception_v2_coco_2017_11_17',
    'faster rcnn resnet':'faster_rcnn_resnet50_coco_2018_01_28'
}
MODEL_NAME = modelname[args.model]
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'models/'+ MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
# Number of classes to detect
NUM_CLASSES = 90

#----------------HUMAN COUNTER-----------------
humancounter = 0
centoroid = []

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Detection
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            # Read frame from camera
            start_time = time.time()
            ret, image_np = cap.read()
            w, h, _ = image_np.shape
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            #Boxes -> ymin, xmin, ymax, xmax
            #Record results
            res_df = res_df.append({'boxes':boxes,'scores':scores,'classes':classes,'num_detections':num_detections}, ignore_index = True)
            # Visualization of the results of a detection.
            image_np, centro = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            
            centoroid.append(centro)
            print('------ {:f} seconds ------'.format(time.time() - start_time))
            # Display output
            cv2.imshow('object detection', cv2.resize(image_np, (target_w, target_h)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

#Store log
if args.save_bool:
    res_df.to_csv('log.csv',index = True, header = True)
print(centoroid)