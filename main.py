import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
import sys
import time
from PIL import Image, ImageDraw, ImageFont

# =====================
# MODE
# =====================
MODE = sys.argv[1] if len(sys.argv) > 1 else "EN"

# =====================
# SPEAK
# =====================
engine = pyttsx3.init()
def speak(text):
    if text.strip():
        engine.say(text)
        engine.runAndWait()

# =====================
# FONT
# =====================
FONT_PATH = "C:/Windows/Fonts/Nirmala.ttc"
font_word  = ImageFont.truetype(FONT_PATH, 48)
font_small = ImageFont.truetype(FONT_PATH, 24)
font_pred  = ImageFont.truetype(FONT_PATH, 36)

def put_text(frame, text, pos, font, color=(255,255,255), anchor="lt"):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color, anchor=anchor)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =====================
# LOAD MODEL
# =====================
model = pickle.load(open("model.pkl", "rb"))

# =====================
# MEDIAPIPE
# =====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,   # raised from 0.7
    min_tracking_confidence=0.7     # raised from 0.6
)
mp_draw = mp.solutions.drawing_utils

# =====================
# NORMALIZATION  ← KEY FIX
# Normalize by wrist + hand span so distance from camera doesn't matter
# =====================
def normalize_landmarks(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

    # Translate to wrist origin
    coords -= coords[0]

    # Scale by max distance from wrist (hand size normalization)
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale > 0:
        coords /= scale

    return coords.flatten().tolist()   # 42 features (21 × x,y)

# =====================
# STATE
# =====================
sentence     = ""
current_word = ""
last_pred    = ""
stable_count = 0

CONFIRM      = 12          # frames the gesture must stay stable  ← increased
COOLDOWN_SEC = 1.2         # seconds to wait before next letter   ← NEW
last_added   = 0.0         # timestamp of last added letter

def is_hindi(ch):
    return any('\u0900' <= c <= '\u097F' for c in ch)

def get_command(x):
    x = x.upper()
    if "SPACE"     in x: return "SPACE"
    if "BACKSPACE" in x: return "BACKSPACE"
    if "SPEAK"     in x: return "SPEAK"
    return None

# =====================
# CAMERA
# =====================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Progress bar config
BAR_X, BAR_Y, BAR_W, BAR_H = 20, 430, 300, 20

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    raw_pred = ""

    # =====================
    # HAND DETECTION
    # =====================
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Use improved normalizer
            data     = normalize_landmarks(hand)
            raw_pred = str(model.predict([data])[0])

    # =====================
    # STABILITY LOGIC  ← FIXED
    # =====================
    if raw_pred == last_pred and raw_pred != "":
        stable_count += 1
    else:
        stable_count = 0
        last_pred    = raw_pred

    # Progress bar (green fill shows stability)
    progress = int(BAR_W * min(stable_count / CONFIRM, 1.0))
    cv2.rectangle(frame, (BAR_X, BAR_Y),
                  (BAR_X + BAR_W, BAR_Y + BAR_H), (50,50,50), -1)
    cv2.rectangle(frame, (BAR_X, BAR_Y),
                  (BAR_X + progress, BAR_Y + BAR_H), (0,255,0), -1)
    cv2.rectangle(frame, (BAR_X, BAR_Y),
                  (BAR_X + BAR_W, BAR_Y + BAR_H), (255,255,255), 1)
    frame = put_text(frame, "Hold gesture...", (BAR_X, BAR_Y - 25),
                     font_small, (200,200,200))

    # =====================
    # APPLY WHEN STABLE + COOLDOWN  ← KEY FIX
    # =====================
    now = time.time()
    if stable_count >= CONFIRM and (now - last_added) >= COOLDOWN_SEC:

        cmd = get_command(raw_pred)

        # Language filter
        if MODE == "EN":
            if raw_pred not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and cmd is None:
                raw_pred = ""
        else:
            if not is_hindi(raw_pred) and cmd is None:
                raw_pred = ""

        if cmd:
            if cmd == "SPACE":
                if current_word:
                    sentence    += current_word + " "
                    current_word = ""
            elif cmd == "BACKSPACE":
                if current_word:
                    current_word = current_word[:-1]
                elif sentence:
                    sentence = sentence[:-1]
            elif cmd == "SPEAK":
                speak(sentence + current_word)

        elif raw_pred:
            current_word += raw_pred

        last_added   = now          # reset cooldown timer
        stable_count = 0            # reset counter AFTER action

    # =====================
    # UI
    # =====================
    full = sentence + current_word

    # Current prediction shown live (even before confirmed)
    if raw_pred and raw_pred not in ("SPACE","BACKSPACE","SPEAK"):
        frame = put_text(frame, f"→ {raw_pred}",
                         (w - 120, 20), font_pred, (0, 255, 100))

    # Main sentence display
    frame = put_text(frame,
                     full if full else "Show Gesture...",
                     (w//2, h//2),
                     font_word,
                     (255, 255, 255),
                     anchor="mm")

    mode_text = "ENGLISH MODE" if MODE == "EN" else "HINDI MODE"
    frame = put_text(frame, mode_text, (20, 20), font_small,
                     (0,255,255) if MODE == "EN" else (255,140,0))

    # Help text
    frame = put_text(frame, "ESC=Quit  C=Clear",
                     (w - 200, h - 30), font_small, (150,150,150))

    cv2.imshow("Sign Language", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('c'):
        sentence     = ""
        current_word = ""

cap.release()
cv2.destroyAllWindows()
