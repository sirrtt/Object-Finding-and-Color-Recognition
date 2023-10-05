import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from statistics import mode

import supervision as sv

# Load the model
def load_model():
    model = YOLO("D:\Intern\Object Finding\yolov8x-seg.pt")  
    model.fuse()
    return model

# Predict the frame of the video or image
def predict(model, frame):
    results = model(frame)
    return results

# Detect the color of the object
def detect_color(frame, segs):
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colors = []

    for seg in segs:
        seg = np.array(seg)
        seg = seg.reshape((-1, 2))
        seg = seg.astype(np.int32)

        # Pick pixel values of the segmentation mask
        x = seg[:, 0]
        y = seg[:, 1]
        x_mid = np.mean(x)
        y_mid = np.mean(y)

        x[x>x_mid] -= 5
        x[x<x_mid] += 5
        y[y>y_mid] -= 5
        y[y<y_mid] += 5

        x = np.clip(x, 0, hsvFrame.shape[1] - 1)
        y = np.clip(y, 0, hsvFrame.shape[0] - 1)
        
        # Pick pixel values of the segmentation mask
        pixel_values = hsvFrame[y, x]

        # Detect the color of each pixel
        for pixel_value in pixel_values:
            hue_value = pixel_value[0]
            sat_value = pixel_value[1]
            val_value = pixel_value[2]

            if hue_value == 0 and sat_value == 0 and val_value <= 50: 
                colors.append("BLACK")
            elif hue_value == 0 and sat_value == 0 and val_value <= 255: 
                colors.append("WHITE")
            elif hue_value < 6:
                colors.append("RED")
            elif hue_value < 22:
                colors.append("ORANGE")
            elif hue_value < 33:
                colors.append("YELLOW")
            elif hue_value < 78:
                colors.append("GREEN")
            elif hue_value < 131:
                colors.append("BLUE")
            elif hue_value < 170:
                colors.append("VIOLET")
            else:
                colors.append(None)

    return colors

# Plot the bounding boxes
def plot_bboxes(results, frame, object_need_to_find_color):
    xyxys = []
    segs = []

    # Extract detections for object class
    for result in results[0]:
        class_id = result.boxes.cls.cpu().numpy().astype(int)
        confidence = result.boxes.conf.cpu().numpy()

        if class_id == object_need_to_find_color and confidence >= 0.2:  # The object I want to find
            xyxys.append(result.boxes.xyxy.cpu().numpy().astype(int))
            segs.append(result.masks.xy)

    for i in range(len(xyxys)):
        x1, y1, x2, y2 = xyxys[i][0][0], xyxys[i][0][1], xyxys[i][0][2], xyxys[i][0][3]
        
        colors = detect_color(frame, segs[i])

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        if segs[i] is not None:
            for seg in segs[i]:
                seg = np.array(seg)
                seg = seg.reshape((-1, 2))
                seg = seg.astype(np.int32)
                frame = cv2.polylines(frame, [seg], True, (0, 0, 255), 4)
        text = mode(colors)
        if text:
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
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

    # timeout = time() + 60

    # while time() < timeout:  
    #     ret, frame = cap.read()
    #     assert ret
    
    #     results = predict(model, frame)
    #     frame = plot_bboxes(results, frame, object_need_to_find_color)

    #     fps = cap.get(cv2.CAP_PROP_FPS)
             
    #     cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
    #     cv2.imshow('YOLOv8 Detection', frame)
 
    #     if cv2.waitKey(5) & 0xFF == 27:
    #         break

    # cap.release()
    # cv2.destroyAllWindows()

    imgpath = r"C:\\Users\\ASUS\\Downloads\\Honda Civic Sport_22 white.jpg"
    img = cv2.imread(imgpath)

    results = predict(model, img)
    img = plot_bboxes(results, img, object_need_to_find_color)

    while True:
        cv2.imshow('Color Detection', img)
        cv2.imwrite('abc.jpg', img)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()