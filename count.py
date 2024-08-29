import cv2
from ultralytics import YOLO, solutions
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='yolov8n.pt')
    parser.add_argument('-c', '--count_mode', type=str, default='region')
    args = parser.parse_args()
    return args

def count_Vehicles(parse_args):
    model = parse_args.model
    count_mode = parse_args.count_mode
    model = YOLO(model)
    cap = cv2.VideoCapture("highway.mp4")
    assert cap.isOpened(), "Error reading video file"
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define line points
    if count_mode == 'line':
        points = [(0, 400), (width, 400)]
    elif count_mode == 'region':
        points = [(760, 100), (width-700, 100), (width, 600), (0, 600)]

    video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    counter = solutions.ObjectCounter(
        view_img=True,
        reg_pts=points,
        names=model.names,
        draw_tracks=True,
        line_thickness=2,
    )

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        tracks = model.track(im0, persist=True, show=False, device=0)

        im0 = counter.start_counting(im0, tracks)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    args = parse_args()
    count_Vehicles(args)

