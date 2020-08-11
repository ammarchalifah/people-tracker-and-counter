![repo banner](https://github.com/ammarchalifah/people-tracker-and-counter/blob/master/asset/banner2.jpg)
# People Tracker and Counter
**Update 2020/08/11:** First release of People Tracker and Counter. This version detects people and their gender in a video frame, track them, count total number of people that go up or down, and write an annotated output video. <br>

**People Tracker and Counter** is a program that created as an academic assignment by **Ammar Chalifah**, supported by **Bisa AI** (an educative AI company based in Indonesia) and 
Biomedical Engineering Department of **Insitut Teknologi Bandung**. The aim of this work is to create a working surveillance program utilising Computer Vision to help
retail stores to collect stores' statistics: total number of customer, their gender, and the timestamp of each identified customer. 

Based on my set up and dataset, using `efficientdet_d0_coco17_tpu-32` model from [TensorFlow 2 Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) as the
object detection model, MS COCO label map, and [Anant Singh's gender classifier model](https://github.com/anantSinghCross/gender-classification), I got **multiple object
tracking accuracy (MOTA)** of 50%.

**Current Features**:
- Logging customer data in each frame to a CSV log file.
- Writing annotated video (with total count and centroid marker) as an output.
- Identify gender of each customer

![record example](https://github.com/ammarchalifah/people-tracker-and-counter/blob/master/asset/cctvresult.gif)

If there any technical questions related to the project, please post an issue or contact me at:
- 18317018@std.stei.itb.ac.id

## Requirements
### Library
Library requirements listed below:
- tensorflow 2.3.0
- opencv 3.4.2
- dlib 19.20.0
- matplotlib
- numpy
- imutils 0.5.3
- python 3.7.7
### Additional Requirements
- Object detection model (saved in TF2 saved model format). Download one [here.](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
- Gender classifier model (output layer with two nodes with softmax activation, saved in h5 format). Download one [here.](https://github.com/anantSinghCross/gender-classification)
- Retail store CCTV footage. You can download some of them from CAVIAR dataset [here.](http://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/)

## How to Use
1. **Create a virtual environment and install all required packages (use pip or conda for convenience).**
2. **Clone this repository and gather additional requirements. Create new `models` directory inside the **people-tracker-and-counter** directory.**
```
git clone https://github.com/ammarchalifah/people-tracker-and-counter.git
cd people-tracker-and-counter
mkdir models
```
3. **Move the object detection model and the gender classifier model to `people-tracker-and-counter/models`. For example:**
```
people-tracker-and-counter
└───models
    ├──efficientdet_d0_coco17_tpu-32
    |   ├───checkpoint
    |   └───saved_model
    |         └───variables
    └───model.h5
```
4. **Open `people-tracker-and-counter/detection_video.py` with your text editor. Go to **line 116** and create a new item in the dict `modelname`. Put your object
detection model's directory as the value and assign an arbitrary key. Later on, you just have to pass the key as an argument when you call the program via terminal.**
```
modelname = {
    'ssd mobilenet':'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8',
    'centernet':'centernet_hg104_1024x1024_kpts_coco17_tpu-32',
    'faster rcnn inception': 'faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8',
    'faster rcnn resnet101': 'faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8',
    'ssd resnet fpn': 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8',
    'efficientdet':'efficientdet_d0_coco17_tpu-32'
}
```
5. **Go to **line 178** and make sure the gender classifier model name matches your model.**
```
print('[INFO] loading gender classifier model...')
gender_model = load_model('models/model.h5')
g_classes = ['woman', 'man']
print('[INFO] gender classifier model loaded')
```
6. **Put your CCTV video inside `videos` directory.**
7. **Run the program by typing:**
```
python detection_video.py -i videos/PATH-TO-VIDEO
```
8. **If you want more control over the parameters, type:**
```
python detection_video.py -m models/PATH-TO-MODEL -i videos/PATH-TO-VIDEO -f INTEGER -c ['person'] -d INTEGER -l INTEGER -g BOOLEAN -o videos/output.avi
```
  - **m or --model**: path to object detection model
  - **i or --input_path**: path to input video file
  - **f or --skip_frame**: frame skip parameter. This program use object detection + object tracking with correlation filter (**implemented with dlib**). 
  Object tracking only done after object detection. This parameter defines the number of frames to implement tracking after each detection. (default: 20)
  - **c or --classes_to_detect**: classes to detect. Normally, we only detect 'person' class. Pass more classes according to ms coco label map if needed.
  - **d or --distance_threshold**: distance threshold, parameter used to decide whether a centroid of object has the same ID to other object in previous frame or not. (default: 70)
  - **l or --longest_disappear**: longest object's disappearance, parameter used to decide whether we have to delete an ID from our tracked objects or not. (default: 15)
  - **g or --log**: log file. Pass **True** if you want to save your log. (default: True)
  - **o or --output**: output file. (default: videos/output.avi)
## Limitations
- Only working for videos with two possible routes (up or down). For different settings, edit the source code!
- Have not tested on real CCTV stream from DVR and real-time applications.
## References
- [Tensorflow 2 Object Detection API](https://github.com/tensorflow/models)
- [Adrian Rosebrock. (July 23, 2018). Simple Object Tracking with Open CV, Pyimagesearch](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/)
- [Gender Classification by Anant Singh](https://github.com/anantSinghCross/gender-classification)
 
