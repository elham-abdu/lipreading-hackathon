import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

print("="*50)
print("Visualizing Mouth Extraction Results")
print("="*50)

# Path to processed data
processed_path = "processed_data/test"

if not os.path.exists(processed_path):
    print("❌ No processed data found!")
    exit()

# Find all video folders with frames
video_folders = []
for root, dirs, files in os.walk(processed_path):
    for d in dirs:
        folder_path = os.path.join(root, d)
        jpgs = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        if len(jpgs) > 0:
            video_folders.append((folder_path, len(jpgs)))

print(f"Found {len(video_folders)} videos with extracted frames")

# Check 3 random videos
for idx in range(min(3, len(video_folders))):
    folder_path, num_frames = random.choice(video_folders)
    
    print(f"\n📁 Video {idx+1}: {folder_path}")
    print(f"   Total frames: {num_frames}")
    
    # Get first 5 frames
    frames = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])[:5]
    
    # Create figure
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    
    for i, frame_file in enumerate(frames):
        img_path = os.path.join(folder_path, frame_file)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Check if image is all black
        is_black = np.all(img == 0)
        
        axes[i].imshow(img_rgb)
        axes[i].set_title(f"Frame {i}\n{'🔴 BLACK' if is_black else '✅ OK'}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Also check original video to compare
print("\n🔍 Let's check the original video for comparison...")

# Find a sample original video
test_path = "data/test"
test_folders = [f for f in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, f))]

if test_folders:
    sample_folder = random.choice(test_folders)
    sample_videos = [f for f in os.listdir(os.path.join(test_path, sample_folder)) 
                    if f.endswith('.mp4')]
    
    if sample_videos:
        sample_video = os.path.join(test_path, sample_folder, sample_videos[0])
        print(f"\n📹 Original video: {sample_video}")
        
        # Open video and show first frame with face detection
        cap = cv2.VideoCapture(sample_video)
        ret, frame = cap.read()
        
        if ret:
            # Try face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw face rectangle
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Mark mouth region (lower half of face)
                cv2.rectangle(frame_rgb, (x, y + h//2), (x+w, y+h), (255, 0, 0), 2)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(frame_rgb)
            plt.title("Original Video - Green: Face, Blue: Mouth Region")
            plt.axis('off')
            plt.show()
        
        cap.release()

print("\n" + "="*50)
print("If you're seeing black images in the visualization,")
print("but saw mouth regions in the preview, there might be")
print("an issue with the saved files. Let me know what you see!")
print("="*50)