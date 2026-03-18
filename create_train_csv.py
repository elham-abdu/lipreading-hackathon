import os
import pandas as pd
from pathlib import Path

print("="*50)
print("Creating Training CSV")
print("="*50)

# Path to your processed training data
train_processed_path = "processed_data/train"

# Get all video folders and their frame counts
video_data = []

# Walk through the processed_data/train structure
for folder in os.listdir(train_processed_path):
    folder_path = os.path.join(train_processed_path, folder)
    if os.path.isdir(folder_path):
        for video_folder in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video_folder)
            if os.path.isdir(video_path):
                # Count frames
                frames = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
                
                # Create video path in the format expected
                video_id = f"train/{folder}/{video_folder}.mp4"
                
                # For now, use placeholder transcription
                # In a real scenario, you'd have actual transcripts
                video_data.append({
                    'path': video_id,
                    'transcription': 'placeholder text',
                    'num_frames': len(frames)
                })

# Create DataFrame
df = pd.DataFrame(video_data)
print(f"\nFound {len(df)} training videos")

# Save to CSV
df.to_csv('train_data.csv', index=False)
print("Saved to train_data.csv")

# Show sample
print("\nSample entries:")
print(df.head())

# Also create a combined CSV with both train and test
print("\n" + "="*50)
print("Creating combined dataset info")
print("="*50)

# Load test data
test_df = pd.read_csv("data/sample_submission.csv")
print(f"Test videos: {len(test_df)}")

# Combine
combined_df = pd.concat([df, test_df], ignore_index=True)
combined_df.to_csv('all_videos.csv', index=False)
print(f"Combined videos: {len(combined_df)}")
print("Saved to all_videos.csv")