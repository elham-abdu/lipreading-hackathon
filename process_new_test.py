import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

print("="*50)
print("Processing NEW Test Videos (49 videos)")
print("="*50)

# Haar Cascade for mouth detection
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
TARGET_SIZE = (96, 96)

def extract_mouth_from_video(video_path, output_folder):
    """Extract mouth region from video frames"""
    video_name = Path(video_path).stem
    video_output_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mouths = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

        if len(mouths) > 0:
            x, y, w_m, h_m = mouths[0]
            y_new = y + h_m // 4
            h_new = h_m // 2
            mouth_roi = frame[y_new:y_new+h_new, x:x+w_m]
            mouth_roi = cv2.resize(mouth_roi, TARGET_SIZE)
        else:
            mouth_roi = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.uint8)

        frame_filename = os.path.join(video_output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, mouth_roi)
        frame_count += 1

    cap.release()
    return frame_count

# Process new test videos
test_path = "new_data/test"
output_base = "processed_data/new_test"
os.makedirs(output_base, exist_ok=True)

# Get all test videos
test_videos = [f for f in os.listdir(test_path) if f.endswith('.mp4')]
print(f"Found {len(test_videos)} test videos")

# Process each video
processed_videos = []
for video in tqdm(test_videos, desc="Processing videos"):
    video_path = os.path.join(test_path, video)
    video_id = video.replace('.mp4', '')
    
    # Extract mouth frames
    num_frames = extract_mouth_from_video(video_path, output_base)
    
    processed_videos.append({
        'video_id': video_id,
        'filename': video,
        'num_frames': num_frames,
        'processed_folder': os.path.join(output_base, video_id)
    })
    
    print(f"   ✅ {video}: {num_frames} frames")

# Save list
import json
with open('processed_new_test.json', 'w') as f:
    json.dump(processed_videos, f, indent=2)

print(f"\n✅ Processed {len(processed_videos)} test videos")
print("💾 Saved to processed_new_test.json")