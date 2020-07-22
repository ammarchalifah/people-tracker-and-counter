import numpy as np
import collections
import math

#Bounding boxes to centoroid converter
def box_to_centoroid(boxes):
    """Function to convert boxes coordinate to centoroid coordinate.
    Args:
        boxes: numpy array of bounding boxes coordinates
        w: width of video stream
        h: height of video stream
    Returns:
        numpy array of object's coordinates"""
    
    center_x = (boxes[0,:,3] + boxes[0,:,1])/2
    center_y = (boxes[0,:,2] + boxes[0,:,0])/2
    return np.stack((center_x, center_y), axis = -1)

def boxes_to_centoroid_2(boxes, classes, scores, category_index, max_boxes_to_draw=20, min_score_thresh=0.5):
    box_to_color_map = collections.defaultdict(str)
    centoroid = []
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(boxes.shape[0]):
        if max_boxes_to_draw == len(box_to_color_map):
            break
        box = tuple(boxes[i].tolist())
        ymin, xmin, ymax, xmax = box
        centoroid.append(((xmax+xmin)/2, (ymax+ymin)/2))
    return centoroid

#Track IDs
def tracker_initializer(centroids, scores, classes, category_index, min_score = 0.5, class_to_draw = ['person']):
    track_id = []
    max_counter = 0
    for i in range(len(centroids)):
        if scores[i] < min_score or category_index[classes[i]]['name'] not in class_to_draw:
            track_id.append(0)
        else:
            track_id.append(max_counter + 1)
            max_counter += 1
    return np.asarray(track_id), max_counter

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt(abs(x2-x1)*abs(x2-x1) + abs(y2-y1)*abs(y2-y1))

def tracker_updater(centroids, scores, classes, category_index, max_counter, prev_centroid, prev_track_id, min_score = 0.5, classes_to_draw = ['person']):
    track_id = np.zeros(len(centroids), dtype = 'int')
    index_track = []
    index_prev_track = []
    for i in range(len(centroids)):
        if scores[i] > min_score and category_index[classes[i]]['name'] in classes_to_draw:
            index_track.append(i)
    for j in range(len(prev_centroid)):
        if prev_track_id[j] != 0:
            index_prev_track.append(j)
    distance = np.zeros((len(index_track), len(index_prev_track)), dtype = 'float')
    for i in range(len(index_track)):
        for j in range(len(index_prev_track)):
            distance[i,j] = euclidean_distance(centroids[index_track[i]][0], centroids[index_track[i]][1], prev_centroid[index_prev_track[j]][0], prev_centroid[index_prev_track[j]][1])
    if distance.shape[0] == 0 or distance.shape[1] == 0:
        return prev_track_id, max_counter
    cols = distance.min(axis = 0).argsort()
    rows = distance.argmin(axis = 0)[cols]
    #------flags------
    print(len(cols))
    print(len(rows))
    print(distance.shape)
    #-----flags---------
    same_index = []
    for idx in range(len(index_prev_track)):
        if distance[rows[idx], cols[idx]]<0.15:
            track_id[index_track[rows[idx]]] = prev_track_id[index_prev_track[cols[idx]]]
            same_index.append(rows[idx])
    for idx in list(set(index_track) - set(same_index)):
        track_id[idx] = max_counter + 1
        max_counter += 1
    return track_id, max_counter