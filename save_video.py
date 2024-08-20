from ultralytics import YOLO
import numpy as np
import cv2
import time

def save_video(path, start_second, end_second):
    cap = cv2.VideoCapture(path) 
    videoFormat = cv2.VideoWriter_fourcc(*'mp4v')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter("output.mp4", videoFormat, 30, (width, height))
    
    fps = cap.get(cv2.CAP_PROP_FPS)    
    start_frame = fps*start_second
    end_frame = fps*end_second
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) #skip offet

    while True:
        ret, frame = cap.read()
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if not ret or current_frame > end_frame:
            break
        output.write(frame)
    

    cap.release()
    output.release()


if __name__ == "__main__":
    save_video("./test_video.mp4", 10, 20)
