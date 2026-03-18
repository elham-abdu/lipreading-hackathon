import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

print("="*50)
print("STEP 2: Mouth Extraction (Training Data)")
print("="*50)

# Haar Cascade for mouth detection
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

TARGET_SIZE = (96, 96)  # Output mouth frame size

def extract_mouth_from_video(video_path, output_folder):
    """
    Extract mouth region from each frame of a video
    """
    video_name = Path(video_path).stem
    video_output_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    mouth_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mouths = haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)
        )

        mouth_roi = None

        if len(mouths) > 0:
            # Take first detected mouth
            x, y, w_m, h_m = mouths[0]

            # Refine: lower half of detected rectangle
            y_new = y + h_m // 4
            h_new = h_m // 2
            mouth_roi = frame[y_new:y_new+h_new, x:x+w_m]

        # If no mouth detected, use black frame
        if mouth_roi is None or mouth_roi.size == 0:
            mouth_roi = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.uint8)

        # Resize to target size
        mouth_roi = cv2.resize(mouth_roi, TARGET_SIZE)

        # Save frame
        frame_filename = os.path.join(video_output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, mouth_roi)

        mouth_frames.append(mouth_roi)
        frame_count += 1

    cap.release()
    return len(mouth_frames)


def process_dataset(split, num_samples=None):
    """
    Process all videos in a dataset split
    """
    print(f"\n📁 Processing {split} videos...")

    split_path = os.path.join("data", split)
    output_base = os.path.join("processed_data", split)
    os.makedirs(output_base, exist_ok=True)

    video_folders = [f for f in os.listdir(split_path) 
                     if os.path.isdir(os.path.join(split_path, f))]

    if num_samples:
        video_folders = video_folders[:num_samples]

    print(f"Found {len(video_folders)} video folders")
    all_videos = []

    for folder in tqdm(video_folders, desc=f"Processing {split}"):
        folder_path = os.path.join(split_path, folder)
        videos = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

        for video in videos:
            video_path = os.path.join(folder_path, video)
            video_id = f"{folder}/{video.replace('.mp4', '')}"

            num_frames = extract_mouth_from_video(
                video_path, 
                os.path.join(output_base, folder)
            )

            all_videos.append({
                'video_id': video_id,
                'path': video_path,
                'num_frames': num_frames,
                'processed_folder': os.path.join(output_base, folder, video.replace('.mp4', ''))
            })

    return all_videos


# --- Main ---
print("\n📄 Loading transcripts...")
df = pd.read_csv("data/sample_submission.csv")
print(f"Loaded {len(df)} entries")

# Process training videos (start with 5 to test)
print("\n🔬 Processing training videos (5 samples)...")
train_videos = process_dataset("train", num_samples=5)

print("\n" + "="*50)
print(f"✅ Processed {len(train_videos)} training videos")
print("="*50)
print("\n📊 Sample of processed videos:")
for v in train_videos[:3]:
    print(f"   {v['video_id']}: {v['num_frames']} frames")

# Save the list for later use
import json
with open('processed_train_videos.json', 'w') as f:
    json.dump(train_videos, f, indent=2)
print("\n💾 Saved video list to processed_train_videos.json")