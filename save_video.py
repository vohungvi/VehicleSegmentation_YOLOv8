import cv2
import time
from datetime import datetime
import os
from ultralytics import YOLO

def get_filename():
    now = datetime.now()
    timestamp = now.strftime("%y-%m-%d_%Hh-%Mm-%Ss")
    return f"{timestamp}.mp4"

def start_recording(video_format, width, height, fps):
    output_path = get_filename()
    output = cv2.VideoWriter(output_path, video_format, fps, (width, height))
    print(f"Đã bắt đầu ghi...{output_path}")
    return output, output_path

def stop_recording(output, output_path):
    output.release()
    new_output_path = os.path.splitext(output_path)[0] + '_' + os.path.splitext(get_filename())[0] + ".mp4"
    os.rename(output_path, new_output_path)
    print(f"Đã dừng ghi và video được lưu dưới dạng {new_output_path}")

def press_key(key, recording, output, output_path, video_format, frame, width, height, fps):
    if key == ord('s') and not recording:
        recording = True
        cv2.putText(frame, 'Recording', (10, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (66, 66, 255), thickness=2)
        output, output_path = start_recording(video_format, width, height, fps)
    if key == ord('e') and recording:
        stop_recording(output, output_path)
        recording = False
    if recording:   
        output.write(frame)
        pass
    quit_flag = key == ord('q')
    return recording, output, output_path, quit_flag

def predict(frame):
    model = YOLO("yolov8n.pt")
    results = model(frame, device=0)
    return results

def draw_bbox(results, frame):
    if results: 
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())  
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return frame
            
def record_video(path):
    cap = cv2.VideoCapture(path)
    video_format = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))    
    
    recording = False
    output = None
    output_path = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = predict(frame)
        frame = draw_bbox(results, frame)
        cv2.imshow("Press 's' to Start, 'e' to Stop, 'q' to Quit", frame)
        key = cv2.waitKey(1) & 0xFF
        recording, output, output_path, quit_flag = press_key(key, recording, output, output_path, video_format, frame, width, height, fps)

        if quit_flag:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # path = 'rtsp://admin:DBTNZR@192.168.1.88:554/ch01/0'
    path = 0
    record_video(path)
    

