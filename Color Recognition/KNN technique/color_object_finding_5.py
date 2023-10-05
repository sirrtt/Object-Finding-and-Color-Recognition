import torch
import numpy as np
import cv2
import os
import os.path
import sys
import csv
import random
import math
import operator
import matplotlib.pyplot as plt
import supervision as sv
import color_histogram_feature_extraction
import knn_classifier

from time import time
from ultralytics import YOLO
from statistics import mode
from scipy import stats

# Load the model
def load_model():
    model = YOLO("D:\Intern\Object Finding\yolov8x.pt")  
    model.fuse()
    return model

# Predict the frame of the video or image
def predict(model, frame, object_need_to_find_color):
    results = model(source = frame, conf = 0.2, classes = object_need_to_find_color)
    return results

def detect_color(frame, x1, y1, x2, y2):
    frame = frame[y1:y2, x1:x2]

    color_histogram_feature_extraction.color_histogram_of_test_image(frame)
    prediction = knn_classifier.main('training.data', 'test.data')
    print('Detected color is:', prediction)
    return str(prediction)

# Plot the bounding boxes
def plot_bboxes(results, frame):
    xyxys = []

    # Extract detections for object class
    for result in results[0]:
        xyxys.append(result.boxes.xyxy.cpu().numpy().astype(int))

    for i in range(len(xyxys)):
        x1, y1, x2, y2 = xyxys[i][0][0], xyxys[i][0][1], xyxys[i][0][2], xyxys[i][0][3]
        
        colors = detect_color(frame, x1, y1, x2, y2)

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        if colors:
            cv2.putText(frame, colors, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Can not detect color", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame	

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using Device: ", device)

    list_of_objects = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"]

    object_need_to_find_color = input('The object that you want to find the color: ')
    object_need_to_find_color = list_of_objects.index(object_need_to_find_color)

    model = load_model()

    cap = cv2.VideoCapture(0)
    assert cap.isOpened()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    imgpath = r"C:\\Users\\ASUS\\Downloads\\18015-MC20BluInfinito-scaled-e1666008987698.jpg"
    img = cv2.imread(imgpath)

    # checking whether the training data is ready
    PATH = 'D:\\Intern\\Object Finding\\training.data'

    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        print ('training data is ready, classifier is loading...')
    else:
        print ('training data is being created...')
        open('D:\\Intern\\Object Finding\\training.data', 'w')
        color_histogram_feature_extraction.training()
        print ('training data is ready, classifier is loading...')

    results = predict(model, img, object_need_to_find_color)
    img = plot_bboxes(results, img)

    while True:
        cv2.imshow('Color Detection', img)
        cv2.imwrite('abc.jpg', img)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()