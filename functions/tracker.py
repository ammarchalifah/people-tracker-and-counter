import numpy as np
import collections

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