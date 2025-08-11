import torch
import numpy as np
import joblib
import json
import cv2
import tempfile
import os
from typing import List, Tuple, Any
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from torchvision import transforms
from models.cnn_model import MY3DCNN
from models.fusion_mlp import FusionMLP
from utils.utils_video import extract_frames
from utils.extract_holistic_feature import extract_holistic_features

LABELS_PATH = "labels_map.json"
MLP_PATH = "fusion_mlp.pth"
CNN_PATH = "trained_3dcnn_epoch23.pth"
SCALER_PATH = "feature_scaler.pkl"
NUM_FRAMES = 16
IMAGE_SIZE = 112

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.Normalize([0.43216, 0.394666, 0.37645],
#                          [0.22803, 0.22145, 0.216989])
# ])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112))
])

def extract_cnn_feature(video_path):
    # Step 1: Extract and preprocess frames
    frames_tensor = extract_frames(video_path, NUM_FRAMES)
    frames_tensor = frames_tensor.unsqueeze(0)

    model = MY3DCNN(num_classes=2185)
    model.load_state_dict(torch.load(CNN_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        feat = model.extract_features(frames_tensor).cpu().numpy()
    return torch.tensor(feat, dtype=torch.float32)

def predict_segment(video_path):
    cnn_feat = extract_cnn_feature(video_path)  # [1, 128]
    holistic_feat = extract_holistic_features(video_path)  # [1, 675]

    # Ensure both are tensors with shape [1, N]
    if cnn_feat.ndim == 1:
        cnn_feat = cnn_feat.unsqueeze(0)

    if isinstance(holistic_feat, np.ndarray):
        holistic_feat = torch.from_numpy(holistic_feat).float()
    if holistic_feat.ndim == 1:
        holistic_feat = holistic_feat.unsqueeze(0)

    # Concatenate
    combined = torch.cat([cnn_feat, holistic_feat], dim=1)

    scaler = joblib.load(SCALER_PATH)
    model = FusionMLP(input_dim=combined.shape[1])
    model.load_state_dict(torch.load(MLP_PATH))
    model.eval()

    scaled = scaler.transform(combined.numpy())
    input_tensor = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    print("CNN feature:", cnn_feat.shape)
    print("Holistic feature:", holistic_feat.shape)
    print("Combined feature:", combined.shape)
    top5 = torch.topk(probs, 5)
    for i in range(5):
        idx = top5.indices[0][i].item()
        label = labels[str(idx)]
        conf = top5.values[0][i].item()
        print(f"{i + 1}. {label} ({conf:.4f})")

    return labels[str(pred.item())], confidence.item()


def hand_displacement(prev_landmarks, curr_landmarks):
    if prev_landmarks is None or curr_landmarks is None:
        return 0
    return np.linalg.norm(np.array(curr_landmarks) - np.array(prev_landmarks))

def extract_hand_landmarks(results) -> np.ndarray:
    def flatten_landmarks(landmarks):
        return [coord for lm in landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

    right = flatten_landmarks(results.right_hand_landmarks) if results.right_hand_landmarks else [0.0] * 63
    left = flatten_landmarks(results.left_hand_landmarks) if results.left_hand_landmarks else [0.0] * 63

    return np.array(right + left, dtype=np.float32)  # Always (126,)

# A higher threshold means only larger hand movements will trigger a new segment.
# Larger window_size smooths out noise in motion detection.
# min_segment_len filters out very short segments (e.g., small hand twitches).
# 1. threshold (in detect_motion_segments)
# Default: 0.02
#
# Try values like:
# 0.005, 0.01, 0.015, 0.025
#
# Lower values = more sensitive to small movements
#
# If it’s too low, you’ll get false positives
#
# If it’s too high, signs won’t be detected
#
# 2. window_size
# Affects how smooth the displacement averaging is.
#
# Try: 3, 5, 7
#
# Smaller = more responsive but noisy
#
# Larger = smoother but delayed
#
# 3. min_segment_len
# Filters out short accidental motion blips
#
# Try: 5, 8, 10
#
# Too small: false detections
#
# Too big: signs might be missed
def detect_motion_segments(video_path: str, threshold=0.01, window_size=5, min_segment_len=5) -> tuple[
    list[Any], float]:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return [], 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    segments = []
    prev_landmarks = None
    motions = []
    start = None
    frame_idx = 0

    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            if frame is None:
                print(f"[ERROR] Empty frame at index {frame_idx}, skipping.")
                frame_idx += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(rgb)
            curr_landmarks = extract_hand_landmarks(results)

            disp = hand_displacement(prev_landmarks, curr_landmarks)
            motions.append(disp)
            prev_landmarks = curr_landmarks

            # Sliding window average to reduce noise
            if len(motions) >= window_size:
                avg_disp = np.mean(motions[-window_size:])
                if avg_disp > threshold and start is None:
                    start = frame_idx
                elif avg_disp < threshold and start is not None:
                    if frame_idx - start >= min_segment_len:
                        segments.append((start, frame_idx))
                    start = None
                print(f"[DEBUG] Frame {frame_idx}: avg_disp = {avg_disp:.5f}")
                cv2.putText(frame, f"Disp: {avg_disp:.5f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                print(f"[DEBUG] Frame {frame_idx}: not enough data to compute avg_disp")

            # Show frame
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1
    cap.release()
    cv2.destroyAllWindows()
    print(f"[DEBUG] Detected motion segments: {segments}")
    return segments, fps

def segment_and_predict_signs(video_path: str) -> List[Tuple[int, int, str]]:
    print(f"[INFO] Detecting segments in: {video_path}")
    segments, fps = detect_motion_segments(video_path)
    predictions = []

    for i, (start, end) in enumerate(segments):
        start_sec = start / fps
        end_sec = end / fps

        # Create temporary video segment
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        ffmpeg_extract_subclip(video_path, start_sec, end_sec, tmp_path)

        pred_class, confidence = predict_segment(tmp_path)
        predictions.append((start, end, pred_class, round(confidence, 4)))

        os.remove(tmp_path)

    # remove consecutive duplicate predicted signs
    filtered = []
    for i, seg in enumerate(predictions):
        if i == 0 or seg[2] != predictions[i - 1][2]:
            filtered.append(seg)

    return filtered