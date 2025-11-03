import cv2
import mediapipe as mp
import numpy as np
import torch
from sstcn_attention_model import SSTCN_Attention
from utils.sign_dict import SIGN_DICT_NO_ACCENT

NUM_CLASSES = 102

# --- Load mô hình ---
model = SSTCN_Attention(num_classes=NUM_CLASSES).to('cpu')
checkpoint = torch.load("sign_sstcn_attention_model.pth", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# --- Khởi tạo MediaPipe ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)  # webcam

sequence = []  # lưu frame gần nhất
SEQ_LEN = 30   # số frame / 1 chuỗi
NUM_JOINTS = 75
sign_dict = SIGN_DICT_NO_ACCENT
pred = 102
threshold = 0.8

def extract_keypoints(results):
    # Pose
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
    else:
        pose = np.zeros((33, 3))
    
    # Left hand
    if results.left_hand_landmarks:
        left = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
    else:
        left = np.zeros((21, 3))
    
    # Right hand
    if results.right_hand_landmarks:
        right = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
    else:
        right = np.zeros((21, 3))
    
    return np.concatenate([pose, left, right]).flatten()  # shape (225,)

frame_skip = 2
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # --- Lấy keypoints 75 joints ---
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)

    cv2.putText(image, f"{len(sequence)}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    # --- Khi đủ 30 frames ---
    if len(sequence) == SEQ_LEN:
        seq = np.array(sequence)               # shape (30, 225)
        seq = seq.reshape(SEQ_LEN, NUM_JOINTS, 3)  # (30, 75, 3)
        seq = seq.transpose(2, 0, 1)          # (3, 30, 75) -> channels, seq_len, joints
        seq = seq[np.newaxis, ...]            # (1, 3, 30, 75)
        seq = torch.tensor(seq, dtype=torch.float32)


        with torch.no_grad():
            out = model(seq)
            probs = torch.softmax(out, dim=1)
            max_prob, pred_idx = torch.max(probs, dim=1)
            if max_prob.item() >= threshold:
                pred = pred_idx.item()
            else:
                pred = 102

        sequence = []  # reset sau mỗi lần nhận diện
    cv2.putText(image, f"Sign: {sign_dict[pred]}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Sign Recognition", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
