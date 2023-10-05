import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

import supervision as sv

def load_model():
    model = YOLO("D:\Intern\Object Finding\yolov8n.pt")  
    model.fuse()
    return model

def predict(model, frame):
    results = model(frame)
    # result = model(frame, agnostic_nms=True)[0]
    return results

def plot_bboxes(results, frame, model):
    box_annotator = sv.BoxAnnotator(thickness=3, text_thickness=3, text_scale=1.5)

    xyxys = []
    confidences = []
    class_ids = []
        
    # Extract detections for person class
    for result in results[0]:
        class_id = result.boxes.cls.cpu().numpy().astype(int)
            
        if class_id == 2: # The object I want to find
            xyxys.append(result.boxes.xyxy.cpu().numpy().astype(int))
            confidences.append(result.boxes.conf.cpu().numpy())
            class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

    if xyxys:
        for i in range(len(xyxys)):
            x1, y1, x2, y2 = xyxys[i][0][0], xyxys[i][0][1], xyxys[i][0][2], xyxys[i][0][3]
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            text = results[0].names[class_ids[i][0].item()]
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:     
        # Setup detections for visualization
        detections = sv.Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )
            
        # Format custom labels
        labels = [f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, trackerid
        in detections]
            
        # Annotate and display frame
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    return frame

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using Device: ", device)

    model = load_model()

    cap = cv2.VideoCapture(0)
    assert cap.isOpened()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
          
        start_time = time()
            
        ret, frame = cap.read()
        assert ret
    
        results = predict(model, frame)
        frame = plot_bboxes(results, frame, model)

        fps = cap.get(cv2.CAP_PROP_FPS)
        print("fps: ", fps)
             
        cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
        cv2.imshow('YOLOv8 Detection', frame)
 
        if cv2.waitKey(5) & 0xFF == 27:
            break

        if time() - start_time >= 30:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()