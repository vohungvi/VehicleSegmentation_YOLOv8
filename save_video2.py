import cv2 
import numpy as np
import datetime 
from ultralytics import YOLO
import time
import threading

# Load model
model = YOLO("yolov8n.pt")

# predict
def predict(frame):
    results = model(frame, device=0, classes=[0])
    return results

def draw_bbox(results, frame):
    if results: 
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())  
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return frame

# auto record
def auto_recording(cap, target, fps, width, height, timeout):
    record_mode = False
    last_detected_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        results = predict(frame)
        frame = draw_bbox(results, frame)
        detected = len(results[0].boxes) > 0

        current_time = time.time()
        if detected:
            last_detected_time = current_time

        # Start recording if detected
        if detected and not record_mode:
            record_mode = True
            d = datetime.datetime.now()
            target_file = d.strftime('%Y%m%d_%H%M%S') + '.mp4'
            color = (frame.ndim > 2) and (frame.shape[2] > 1)
            target.open(target_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), color)
            print("Automatic recording started.")

        # Stop recording if no detection within the timeout
        if not detected and record_mode and (current_time - last_detected_time > timeout):
            record_mode = False
            target.release()
            print("Automatic recording stopped due to timeout.")

        if record_mode:
            target.write(frame)

        cv2.putText(frame, 'Auto Recording' if record_mode else '', (10, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), thickness=2)
        cv2.imshow('Video Player', frame)

        key = cv2.waitKey(1)
        # Switch to manual mode
        if key == ord('r'):  
            print("Switching to manual recording mode.")
            return False
        if key == ord('q'):  
            return True

def manual_recording(cap, target, fps, width, height):
    record_mode = False
    manual_stop_time = None
    auto_restart_delay = 15  # start auto after delay time

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        results = predict(frame)
        frame = draw_bbox(results, frame)
        current_time = time.time()

        # Auto start recording after delay time of manual stop
        if manual_stop_time and current_time - manual_stop_time >= auto_restart_delay:
            print("Switching to automatic recording mode after 5 minutes of manual stop.")
            return False  # Switch back to auto mode

        key = cv2.waitKey(1)

        # Start manual recording with 'r'
        if key == ord('r') and not record_mode:
            record_mode = True
            manual_stop_time = None
            d = datetime.datetime.now()
            target_file = d.strftime('%Y%m%d_%H%M%S') + '.mp4'
            color = (frame.ndim > 2) and (frame.shape[2] > 1)
            target.open(target_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), color)
            print("Manual recording started.")

        # Stop recording manually with 's'
        if key == ord('s') and record_mode:
            record_mode = False
            target.release()
            manual_stop_time = current_time  # Record the time when 's' is pressed
            print("Recording stopped manually.")

        if record_mode:
            target.write(frame)

        cv2.putText(frame, 'Manual Recording' if record_mode else '', (10, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), thickness=2)
        cv2.imshow('Video Player', frame)

        if key == ord('q'):  # Quit
            break

def record_video(path=None):
    cap = cv2.VideoCapture(path)
    timeout = 5  # no object detected time

    if cap.isOpened():
        target = cv2.VideoWriter()
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while True:
            if auto_recording(cap, target, fps, width, height, timeout):
                break  
            if manual_recording(cap, target, fps, width, height):
                break 
            
        cap.release()
        cv2.destroyAllWindows()

def main():
    path = "rtsp://admin:DBTNZR@192.168.1.88:554/ch01/0"
    record_video(path)

if __name__ == "__main__":
    main()