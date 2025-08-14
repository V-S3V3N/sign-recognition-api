import os
import cv2
import torch
import mediapipe as mp
import numpy as np

def extract_holistic_features(video_path, num_frames=16):
    mp_holistic = mp.solutions.holistic
    mp_hands = mp.solutions.hands

    holistic = mp_holistic.Holistic(static_image_mode=False)
    hands = mp_hands.Hands(static_image_mode=False)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_idxs = np.linspace(0, total - 1, num_frames, dtype=int)

    landmarks = []

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in sample_idxs:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            frame_vec = []

            # Try holistic (pose + hands)
            if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                for part in [results.pose_landmarks, results.left_hand_landmarks, results.right_hand_landmarks]:
                    if part:
                        for lm in part.landmark:
                            frame_vec.extend([lm.x, lm.y, lm.z])
                    else:
                        count = 33 if part == results.pose_landmarks else 21
                        frame_vec.extend([0] * (count * 3))
            else:
                # Fallback to hands only
                hand_result = hands.process(image)
                if hand_result.multi_hand_landmarks:
                    for hand in hand_result.multi_hand_landmarks:
                        for lm in hand.landmark:
                            frame_vec.extend([lm.x, lm.y, lm.z])
                # Pad to length 675
                frame_vec.extend([0] * (675 - len(frame_vec)))

            # Safety: force vector to be exactly 675
            if len(frame_vec) < 675:
                frame_vec.extend([0] * (675 - len(frame_vec)))
            elif len(frame_vec) > 675:
                frame_vec = frame_vec[:675]

            landmarks.append(frame_vec)

    cap.release()
    holistic.close()
    hands.close()

    if len(landmarks) == 0:
        return torch.zeros(675)

    landmarks = torch.tensor(landmarks, dtype=torch.float32)
    return torch.mean(landmarks, dim=0)

