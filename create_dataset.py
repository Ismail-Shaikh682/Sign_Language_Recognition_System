print("Script started!")

import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd
import json

print("All imports ok!")

data         = []
label_map    = {}
total_images = 0
detected     = 0
skipped      = 0

MODES = ["english", "hindi_vowels", "hindi_consonants", "special"]

mp_hands = mp.solutions.hands

# =====================
# SAME NORMALIZER AS 4_predict.py  ← KEY FIX
# Training and inference MUST use identical features
# =====================
def normalize_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
    coords -= coords[0]                                    # wrist origin
    scale   = np.max(np.linalg.norm(coords, axis=1))
    if scale > 0:
        coords /= scale                                    # size normalize
    return coords.flatten().tolist()                       # 42 features

print("="*55)
print("Creating dataset from all collected images...")
print("="*55)

for mode in MODES:
    mode_path = f"dataset/{mode}"
    if not os.path.exists(mode_path):
        print(f"Skipping {mode} — folder not found")
        continue

    folders = [f for f in os.listdir(mode_path)
               if os.path.isdir(f"{mode_path}/{f}")]
    print(f"\nMode: {mode} — {len(folders)} labels found")

    for folder_name in folders:
        folder_path = f"{mode_path}/{folder_name}"

        label_file = f"{folder_path}/label.txt"
        if os.path.exists(label_file):
            with open(label_file, "r", encoding="utf-8") as f:
                actual_label = f.read().strip()
        else:
            actual_label = folder_name

        label_map[folder_name] = actual_label

        images = [f for f in os.listdir(folder_path)
                  if f.lower().endswith(('.jpg', '.png'))]
        print(f"  [{actual_label}]: {len(images)} images", end=" ", flush=True)

        label_detected = 0

        # Higher confidence for cleaner training data
        with mp_hands.Hands(
            static_image_mode=True,
            min_detection_confidence=0.6   # raised from 0.3
        ) as hands_detector:

            for img_name in images:
                img_path = f"{folder_path}/{img_name}"
                img      = cv2.imread(img_path)
                if img is None:
                    skipped += 1
                    continue

                total_images += 1
                found         = False
                result        = None

                # Try original + mild upscale only (downscale hurts quality)
                for scale in [1.0, 1.3]:
                    h_img, w_img = img.shape[:2]
                    resized      = cv2.resize(img,
                                             (int(w_img*scale), int(h_img*scale)))
                    rgb          = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    result       = hands_detector.process(rgb)
                    if result.multi_hand_landmarks:
                        found = True
                        break

                if found and result is not None:
                    for hand in result.multi_hand_landmarks:
                        row = normalize_landmarks(hand)
                        row.append(actual_label)
                        data.append(row)
                    detected      += 1
                    label_detected += 1
                else:
                    skipped += 1

        print(f"→ Detected: {label_detected}/{len(images)}")

print(f"\n{'='*55}")
print(f"Total images    : {total_images}")
print(f"Hand detected   : {detected}")
print(f"Skipped         : {skipped}")
detection_rate = (detected / total_images * 100) if total_images > 0 else 0
print(f"Detection rate  : {detection_rate:.1f}%")
print(f"Total rows      : {len(data)}")

if len(data) == 0:
    print("ERROR: No data collected! Check your dataset/ folder.")
else:
    df = pd.DataFrame(data)
    df.to_csv("dataset.csv", index=False)
    print("\ndataset.csv saved!")

    with open("label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print("label_map.json saved!")
    print("\nNext → Run 3_train_model.py")
