import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
import time
import tempfile
from pathlib import Path

FRAMES = 16
IMAGE_SIZE = 112
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

def extract_frames(path, num_frames=FRAMES):
    """
    Extract frames with robust temporary file handling
    """
    # Debug: Print file info
    print(f"Attempting to open video: {path}")
    print(f"File exists: {os.path.exists(path)}")
    if os.path.exists(path):
        print(f"File size: {os.path.getsize(path)} bytes")

    # Wait a moment for file system to sync (especially important on Windows)
    time.sleep(0.1)

    # Try multiple times with small delays
    cap = None
    max_retries = 3

    for attempt in range(max_retries):
        try:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                break
            else:
                print(f"Attempt {attempt + 1}: Failed to open video file")
                if cap:
                    cap.release()
                time.sleep(0.2)  # Wait before retry
        except Exception as e:
            print(f"Attempt {attempt + 1}: Exception opening video: {e}")
            if cap:
                cap.release()
            time.sleep(0.2)

    # Final check
    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        raise ValueError(f"Cannot open video file after {max_retries} attempts: {path}")

    # Get video properties
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video opened successfully: {total} frames, {fps} fps, {width}x{height}")

    if total <= 0:
        cap.release()
        raise ValueError(f"Video file appears to be empty: {path}")

    if total < num_frames:
        indices = list(range(total)) + [total - 1] * (num_frames - total)
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if ret and frame is not None:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = transform(frame_rgb)
                frames.append(frame_tensor)
            except Exception as e:
                print(f"Error processing frame {idx}: {e}")
                # Use fallback
                if frames:
                    frames.append(frames[-1].clone())
                else:
                    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    black_tensor = transform(black_frame)
                    frames.append(black_tensor)
        else:
            print(f"Warning: Could not read frame at index {idx}")
            if frames:
                frames.append(frames[-1].clone())
            else:
                black_frame = np.zeros((height if height > 0 else 224, width if width > 0 else 224, 3), dtype=np.uint8)
                black_tensor = transform(black_frame)
                frames.append(black_tensor)

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames could be extracted from the video")

    # Ensure we have the requested number of frames
    while len(frames) < num_frames:
        frames.append(frames[-1].clone())

    return torch.stack(frames, dim=1)  # Shape: [C, T, H, W]