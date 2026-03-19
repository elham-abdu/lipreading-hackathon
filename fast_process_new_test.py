import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import multiprocessing
import time

print("="*50)
print("Processing NEW Test Videos (Fixed Version)")
print("="*50)

# Haar Cascade for mouth detection
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
TARGET_SIZE = (96, 96)

def extract_mouth_from_video(video_info):
    """Extract mouth region from video frames"""
    video_path, output_base, video_id = video_info
    
    video_output_folder = os.path.join(output_base, video_id)
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
    return (video_id, frame_count)

def process_videos_sequential(video_infos):
    """Process videos sequentially (no multiprocessing)"""
    results = []
    for video_info in tqdm(video_infos, desc="Processing videos"):
        result = extract_mouth_from_video(video_info)
        results.append(result)
        print(f"   ✅ {result[0]}.mp4: {result[1]} frames")
    return results

def main():
    # Get all test videos
    test_path = "new_data/test"
    output_base = "processed_data/new_test"
    os.makedirs(output_base, exist_ok=True)

    test_videos = [f for f in os.listdir(test_path) if f.endswith('.mp4')]
    print(f"Found {len(test_videos)} test videos")

    # Prepare video info list
    video_infos = []
    for video in test_videos:
        video_path = os.path.join(test_path, video)
        video_id = video.replace('.mp4', '')
        video_infos.append((video_path, output_base, video_id))

    # Process sequentially (faster than fixing multiprocessing)
    print("\n🚀 Starting sequential processing...")
    start_time = time.time()
    
    results = process_videos_sequential(video_infos)
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Processed {len(results)} videos in {elapsed_time/60:.1f} minutes!")

    # Collect results
    processed_videos = []
    for video_id, num_frames in results:
        processed_videos.append({
            'video_id': video_id,
            'filename': f"{video_id}.mp4",
            'num_frames': num_frames,
            'processed_folder': os.path.join(output_base, video_id)
        })

    # Save list
    import json
    with open('processed_new_test.json', 'w') as f:
        json.dump(processed_videos, f, indent=2)

    print("💾 Saved to processed_new_test.json")
    
    # Show summary
    print(f"\n📊 Summary:")
    print(f"   Total videos: {len(processed_videos)}")
    print(f"   Total frames: {sum(v['num_frames'] for v in processed_videos)}")
    print(f"   Avg frames/video: {sum(v['num_frames'] for v in processed_videos)/len(processed_videos):.1f}")

if __name__ == '__main__':
    main()