import cv2
import os
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np


FONT_PATH = "C:/Windows/Fonts/Nirmala.ttc"
try:
    font_large = ImageFont.truetype(FONT_PATH, 42)
    font_medium = ImageFont.truetype(FONT_PATH, 28)
    font_small = ImageFont.truetype(FONT_PATH, 22)
    print("Font loaded!")
except:
    print("ERROR: Font not found!")
    exit()

def put_text(frame, text, position, font, color=(0, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =====================
# LABELS (SEPARATED CLEANLY)
# =====================

ENGLISH_LABELS = [(chr(i), chr(i)) for i in range(65, 91)]  # A-Z

HINDI_VOWELS = [
    ("vowel_0","अ"),("vowel_1","आ"),("vowel_2","इ"),
    ("vowel_3","ई"),("vowel_4","उ"),("vowel_5","ऊ"),
    ("vowel_6","ए"),("vowel_7","ऐ"),("vowel_8","ओ"),
    ("vowel_9","औ"),("vowel_10","अं"),("vowel_11","अः"),
]

HINDI_CONSONANTS = [
    ("ka","क"),("kha","ख"),("ga","ग"),("gha","घ"),("nga","ङ"),
    ("cha","च"),("chha","छ"),("ja","ज"),("jha","झ"),("nya","ञ"),
    ("tta","ट"),("ttha","ठ"),("dda","ड"),("ddha","ढ"),("nna","ण"),
    ("ta","त"),("tha","थ"),("da","द"),("dha","ध"),("na","न"),
    ("pa","प"),("pha","फ"),("ba","ब"),("bha","भ"),("ma","म"),
    ("ya","य"),("ra","र"),("la","ल"),("va","व"),
    ("sha","श"),("ssha","ष"),("sa","स"),("ha","ह"),
]

SPECIAL = [
    ("SPACE","SPACE"),
    ("BACKSPACE","BACKSPACE"),
    ("SPEAK","SPEAK"),
]

# =====================
# 🔥 MODE SELECT (IMPORTANT)
# =====================
# "english"
# "hindi_vowels"
# "hindi_consonants"
# "special"

MODE = "special"


if MODE == "english":
    LABELS = ENGLISH_LABELS
    LANGUAGE_DISPLAY = "ENGLISH ONLY"
elif MODE == "hindi_vowels":
    LABELS = HINDI_VOWELS
    LANGUAGE_DISPLAY = "HINDI VOWELS"
elif MODE == "hindi_consonants":
    LABELS = HINDI_CONSONANTS
    LANGUAGE_DISPLAY = "HINDI CONSONANTS"
elif MODE == "special":
    LABELS = SPECIAL
    LANGUAGE_DISPLAY = "SPECIAL COMMANDS"
else:
    print("Invalid MODE!")
    exit()

# =====================
IMAGES_PER_LABEL = 150
DELAY = 0.05
# =====================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"\n{'='*55}")
print(f"MODE        : {LANGUAGE_DISPLAY}")
print(f"TOTAL SIGNS : {len(LABELS)}")
print(f"{'='*55}\n")

for idx, (safe_label, display_label) in enumerate(LABELS):
    folder = f"dataset/{MODE}/{safe_label}"
    os.makedirs(folder, exist_ok=True)

    with open(f"{folder}/label.txt", "w", encoding="utf-8") as f:
        f.write(display_label)

    count = 0
    ready = False

    print(f"[{idx+1}/{len(LABELS)}] Collecting: {display_label}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # ROI
        cv2.rectangle(display, (80, 105), (560, 435), (0, 255, 0), 2)

        # Mode text
        display = put_text(display, LANGUAGE_DISPLAY, (10, 5), font_small, (255,255,0))

        # Label
        display = put_text(display, f"{display_label}", (10, 40), font_large)

        # Instructions
        display = put_text(display, "Press S to start", (10, 440), font_small, (0,0,255))

        if ready:
            roi = frame[105:435, 80:560]
            cv2.imwrite(f"{folder}/{count}.jpg", roi)
            count += 1
            time.sleep(DELAY)

            display = put_text(display, f"{count}/{IMAGES_PER_LABEL}", (450, 30), font_medium, (0,255,0))

        cv2.imshow("Collect Data", display)

        if count >= IMAGES_PER_LABEL:
            print(f"✔️ Done: {display_label}")
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            ready = True

        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()

print("\n✅ DATA COLLECTION DONE!")
print("Next → Run 2_create_dataset.py")
