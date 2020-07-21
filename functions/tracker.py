import numpy as np

#Bounding boxes to centoroid converter
def box_to_centoroid(boxes):
    """Function to convert boxes coordinate to centoroid coordinate.
    Args:
        boxes: numpy array of bounding boxes coordinates
        w: width of video stream
        h: height of video stream
    Returns:
        numpy array of object's coordinates"""
    
    center_x = (boxes[0,:,3] - boxes[0,:,1])
    center_y = (boxes[0,:,2] - boxes[0,:,0])
    return np.stack((center_x, center_y), axis = -1)