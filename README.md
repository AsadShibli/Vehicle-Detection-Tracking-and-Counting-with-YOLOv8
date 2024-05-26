# Vehicle Detection, Tracking, and Counting with YOLOv8

Welcome to this comprehensive tutorial on using YOLOv8 for real-time vehicle detection and tracking! This repository provides everything you need to get started with identifying and monitoring vehicles using advanced computer vision techniques. Here's what you'll learn:

1.**Real-Time Vehicle Detection with YOLOv8**: Dive into the intricacies of YOLOv8, a state-of-the-art object detection algorithm, and learn how to identify vehicles in real-time.

2.**Vehicle Tracking Fundamentals**: Gain insights into the fundamentals of vehicle tracking, enabling you to monitor vehicle movements seamlessly.

3.**Advanced Techniques for Direction-Wise Vehicle Counting**: Explore advanced techniques to count vehicles based on their direction, offering invaluable insights into traffic flow analysis.


## Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Installation](#installation)
- [Setup](#setup)
- [Code Explanation](#code-explanation)
- [Running the Code](#running-the-code)
- [Results](#results)
- [Conclusion](#conclusion)


## Introduction
This project demonstrates how to use YOLOv8 for vehicle detection, tracking, and counting using OpenCV and Python. The primary objective is to identify vehicles in a video, track their movement, and count the number of vehicles moving in different directions.

## Demo

https://github.com/AsadShibli/Vehicle-Detection-Tracking-and-Counting-with-YOLOv8/assets/119102237/e0a4e947-795b-437c-b29b-322ad5f8fc13

## Installation
First, ensure you have all the necessary dependencies installed. You can install the required libraries using the following commands:

```bash
!pip install ultralytics
!pip install pandas
!pip install opencv-python-headless
```
## Setup
First, ensure you have all the necessary dependencies installed. You can install the required libraries using the following commands:

```bash
!pip install ultralytics
!pip install pandas
!pip install opencv-python-headless
```
1 . **Mount Google Drive**: This is needed if you're using Google Colab.

```python
from google.colab import drive
drive.mount('/content/drive')
```
2. Import Libraries and Initialize YOLO Model:
```python
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker

model = YOLO('yolov8s.pt')

```
3. Class List and Tracker Initialization:
```python
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

tracker = Tracker()
```
## Code Explanation:

Here is the complete code with detailed explanations:
```python
import cv2
import pandas as pd

# Assuming model, tracker, and class_list are already defined
cap = cv2.VideoCapture('/content/drive/MyDrive/temp/2165-155327596_small.mp4')

# Initialize video writer
output_video_folder = "output_video/"
output_video_path = output_video_folder + "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (1020, 500))

count = 0
down = {}
up = {}
counter_down = []
counter_up = []

# Maintain a set of IDs that have already been counted
counted_ids_down = set()
counted_ids_up = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()  # Ensure the data is on CPU
    px = pd.DataFrame(a).astype("float")

    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int((x3 + x4) // 2)
        cy = int((y3 + y4) // 2)

        red_line_y = 255
        blue_line_y = 378
        offset = 10

        # Condition for counting the cars which are entering from red line and exiting from blue line
        if red_line_y < (cy + offset) and red_line_y > (cy - offset):
            down[id] = cy
        if id in down and id not in counted_ids_down:
            if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                counter_down.append(id)  # Append to list of counted IDs
                counted_ids_down.add(id)  # Mark ID as counted

        # Condition for cars entering from blue line
        if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
            up[id] = cy
        if id in up and id not in counted_ids_up:
            if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                counter_up.append(id)  # Append to list of counted IDs
                counted_ids_up.add(id)  # Mark ID as counted

    text_color = (255, 255, 255)  # White color for text
    red_color = (0, 0, 255)  # (B, G, R)
    blue_color = (255, 0, 0)  # (B, G, R)
    green_color = (0, 255, 0)  # (B, G, R)

    cv2.line(frame, (305, 255), (680, 255), red_color, 3)  # Starting coordinates and end of line coordinates
    cv2.putText(frame, 'red line', (305, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.line(frame, (266, 378), (750, 378), blue_color, 3)  # Second line
    cv2.putText(frame, 'blue line', (266, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    downwards = len(counter_down)
    cv2.putText(frame, 'going down - ' + str(downwards), (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color, 1, cv2.LINE_AA)

    upwards = len(counter_up)
    cv2.putText(frame, 'going up - ' + str(upwards), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    # Write the frame to the output video
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

## Running the Code:
1.**Upload the Video**: Ensure your video file is uploaded to the specified directory in your Google Drive.
2.**Execute the Code**: Run the script in Google Colab or your local environment.
3. **Output**: The processed video will be saved in the output_video/ directory with detected and tracked vehicles.

## Results:

The script processes the video, detects vehicles, tracks their movements, and counts the number of vehicles moving up and down across defined lines. The resulting video showcases these detections and counts in real-time.

## Conclusion

This project demonstrates how to effectively use YOLOv8 for real-time vehicle detection, tracking, and counting. These techniques can be applied to various computer vision tasks, providing a solid foundation for further exploration and application in the field of computer vision.

