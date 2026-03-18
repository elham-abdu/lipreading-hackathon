import os
import cv2
import matplotlib.pyplot as plt
import random

print("="*50)
print("Checking Mouth Extraction Quality")
print("="*50)

processed_path = "processed_data/test"

if not os.path.exists(processed_path):
    print("❌ No processed data found! Run step2_extract_mouth.py first")
    exit()

video_folders = [f for f in os.listdir(processed_path) if os.path.isdir(os.path.join(processed_path, f))]
print(f"Found {len(video_folders)} processed videos")

# Pick a random video folder with frames
valid_folders = []
for folder in video_folders:
    folder_path = os.path.join(processed_path, folder)
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    for sub in subfolders:
        sub_path = os.path.join(folder_path, sub)
        frames = [f for f in os.listdir(sub_path) if f.endswith(".jpg")]
        if frames:
            valid_folders.append(sub_path)

if not valid_folders:
    print("❌ No frames found in any folder!")
    exit()

# Randomly pick one folder
folder_path = random.choice(valid_folders)
frames = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
print(f"\n📁 Checking video: {folder_path}")
print(f"   Total frames: {len(frames)}")

# Show first 5 frames
fig, axes = plt.subplots(1, min(5, len(frames)), figsize=(15, 3))
for i in range(min(5, len(frames))):
    img = cv2.imread(os.path.join(folder_path, frames[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[i].imshow(img)
    axes[i].set_title(f"Frame {i}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()
