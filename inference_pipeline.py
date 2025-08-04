import torch
import numpy as np
import joblib
import json

from torchvision import transforms
from models.cnn_model import MY3DCNN
from models.fusion_mlp import FusionMLP
from utils.utils_video import extract_frames
from utils.extract_holistic_feature import extract_holistic_features

LABELS_PATH = "labels_map.json"
MLP_PATH = "fusion_mlp.pth"
CNN_PATH = "trained_3dcnn.pth"
SCALER_PATH = "feature_scaler.pkl"
NUM_FRAMES = 16
IMAGE_SIZE = 112

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize([0.43216, 0.394666, 0.37645],
                         [0.22803, 0.22145, 0.216989])
])

def predict_sign(video_path: str) -> str:
    device = torch.device("cpu")

    # Step 1: Extract and preprocess frames
    frames_tensor = extract_frames(video_path, NUM_FRAMES)
    frames_tensor = frames_tensor.unsqueeze(0)

    # Step 2: 3D-CNN feature
    cnn_model = MY3DCNN(num_classes=2185).to(device)
    cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=device))
    cnn_model.eval()
    with torch.no_grad():
        cnn_feature = cnn_model.extract_features(frames_tensor).cpu().numpy()

    # Step 3: MediaPipe landmarks
    landmark_feature = extract_holistic_features(video_path)
    if isinstance(landmark_feature, torch.Tensor):
        landmark_feature = landmark_feature.view(1, -1).cpu().numpy()
    else:
        landmark_feature = landmark_feature.reshape(1, -1)

    # Ensure cnn_feature and landmark_feature have correct shape
    if cnn_feature.ndim == 1:
        cnn_feature = np.expand_dims(cnn_feature, axis=0)
    if landmark_feature.ndim == 1:
        landmark_feature = np.expand_dims(landmark_feature, axis=0)

    # Step 4: Combine and scale
    combined_feature = np.concatenate([cnn_feature, landmark_feature], axis=1)  # [1, 803]
    scaler = joblib.load(SCALER_PATH)
    scaled_feature = scaler.transform(combined_feature)

    # Step 5: MLP prediction
    mlp = FusionMLP(input_dim=scaled_feature.shape[1])
    mlp.load_state_dict(torch.load(MLP_PATH))
    mlp.eval()
    with torch.no_grad():
        output = mlp(torch.tensor(scaled_feature, dtype=torch.float32))
        pred_class = torch.argmax(output, dim=1).item()

    # Step 6: Map label
    with open(LABELS_PATH, "r") as f:
        idx_to_label = json.load(f)

    return idx_to_label[str(pred_class)]
