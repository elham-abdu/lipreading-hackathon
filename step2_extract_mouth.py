import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

print("="*50)
print("STEP 2: Mouth Extraction (ALL Test Videos)")
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
    total_videos_processed = 0

    for folder in tqdm(video_folders, desc=f"Processing {split}"):
        folder_path = os.path.join(split_path, folder)
        videos = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

        for video in videos:
            video_path = os.path.join(folder_path, video)
            video_id = f"{folder}/{video.replace('.mp4', '')}"
            
            # Check if already processed
            processed_folder = os.path.join(output_base, folder, video.replace('.mp4', ''))
            if os.path.exists(processed_folder) and len(os.listdir(processed_folder)) > 0:
                # Count existing frames
                num_frames = len([f for f in os.listdir(processed_folder) if f.endswith('.jpg')])
                print(f"   ⏩ Already processed: {video_id} ({num_frames} frames)")
            else:
                # Process new video
                num_frames = extract_mouth_from_video(video_path, os.path.join(output_base, folder))
                total_videos_processed += 1

            all_videos.append({
                'video_id': video_id,
                'path': video_path,
                'num_frames': num_frames,
                'processed_folder': processed_folder
            })
            
            # Progress update every 100 videos
            if len(all_videos) % 100 == 0:
                print(f"   📊 Progress: {len(all_videos)} videos indexed")

    print(f"\n📊 Total videos in {split}: {len(all_videos)}")
    print(f"   Newly processed: {total_videos_processed}")
    return all_videos


# --- Main ---
print("\n📄 Loading transcripts...")
df = pd.read_csv("data/sample_submission.csv")
print(f"Loaded {len(df)} entries")

# Ask user what to process
print("\n🔧 What would you like to process?")
print("   1. Process ALL test videos (3000 videos - takes 2-3 hours)")
print("   2. Process a small sample of test videos (faster)")
print("   3. Process more training videos")

choice = input("\nEnter your choice (1, 2, or 3): ").strip()

if choice == '1':
    print("\n🚀 Processing ALL test videos...")
    test_videos = process_dataset("test", num_samples=None)
    
    print("\n" + "="*50)
    print(f"✅ Processed {len(test_videos)} test videos total!")
    print("="*50)
    
    # Save the list
    with open('processed_test_videos.json', 'w') as f:
        json.dump(test_videos, f, indent=2)
    print("\n💾 Saved video list to processed_test_videos.json")
    
    # Show sample
    print("\n📊 Sample of processed videos:")
    for v in test_videos[:5]:
        print(f"   {v['video_id']}: {v['num_frames']} frames")

elif choice == '2':
    sample_size = int(input("How many test videos to process? (e.g., 10): "))
    print(f"\n🔬 Processing {sample_size} test videos...")
    test_videos = process_dataset("test", num_samples=sample_size)
    
    print("\n" + "="*50)
    print(f"✅ Processed {len(test_videos)} test videos")
    print("="*50)

elif choice == '3':
    num_train = int(input("How many MORE training videos to process? (e.g., 20): "))
    print(f"\n🔬 Processing {num_train} additional training videos...")
    train_videos = process_dataset("train", num_samples=num_train)
    
    print("\n" + "="*50)
    print(f"✅ Processed {len(train_videos)} training videos")
    print("="*50)
    
    # Update the combined CSV
    print("\n📊 Updating combined CSV...")
    all_videos_df = pd.read_csv('all_videos.csv')
    
    # Add new training videos if they don't exist
    existing_paths = set(all_videos_df['path'])
    new_videos = []
    for v in train_videos:
        path = f"train/{v['video_id']}.mp4"
        if path not in existing_paths:
            new_videos.append({
                'path': path,
                'transcription': 'placeholder text',
                'num_frames': v['num_frames']
            })
    
    if new_videos:
        new_df = pd.DataFrame(new_videos)
        updated_df = pd.concat([all_videos_df, new_df], ignore_index=True)
        updated_df.to_csv('all_videos.csv', index=False)
        print(f"✅ Added {len(new_videos)} new videos to all_videos.csv")
    else:
        print("✅ No new videos to add")

else:
    print("❌ Invalid choice")

print("\n" + "="*50)
print("✅ Step 2 complete!")
print("="*50)