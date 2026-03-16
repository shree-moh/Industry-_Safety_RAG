import cv2
import os

def extract_frames_from_video(video_path, output_dir, seconds_interval=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 1
    frame_interval = max(1, int(fps * seconds_interval))
    basename = os.path.splitext(os.path.basename(video_path))[0]
    frame_no = 0
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % frame_interval == 0:
            out_name = f"{basename}_frame{frame_no}.jpg"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, frame)
            count += 1
        frame_no += 1
    cap.release()
    print(f"{video_path}: {count} frames extracted.")

# Usage
video_input_folder = r"/Users/srimanmohapatra/Downloads/occupational_safety_/data/Video_Input"
video_output_folder = r"/Users/srimanmohapatra/Downloads/occupational_safety_/output/video_frames"

for fname in os.listdir(video_input_folder):
    if fname.lower().endswith((".mp4", ".avi", ".mov")):
        extract_frames_from_video(os.path.join(video_input_folder, fname), video_output_folder, seconds_interval=1)
