import cv2
import pyttsx3
import mediapipe as mp
from google import genai
from google.genai import types
import numpy as np
import time
import threading
import io
from PIL import Image
from collections import deque

# ─────────────────────────────────────────────────────────────
# 🔑  PROJECT CONFIG
# ─────────────────────────────────────────────────────────────


GEMINI_API_KEY = "AIzaSyAp0yz5hKlSCLAbSj-4Aal9FIT8eBadHbw" 
GEMINI_MODEL = "gemini-3-flash-preview"

# Limits optimized for 2026 Free Tier (Safe: ~14 requests per minute)
CAPTURE_INTERVAL   = 2.5    # Give the API room to breathe

HOLD_SECONDS       = 1.2    
TTS_RATE = 165          

# ─────────────────────────────────────────────────────────────
# AI & SPEECH ENGINE
# ─────────────────────────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)
engine = pyttsx3.init()
engine.setProperty('rate', TTS_RATE)

def speak(text):
    def _run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()

# ─────────────────────────────────────────────────────────────
# HAND TRACKING
# ─────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.7
)

# ─────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────
current_word = []
final_sentence = []
history_log = deque(["System Online"], maxlen=8)
gemini_busy = False
last_call_time = 0
last_sign = ""

def process_sign(roi_img):
    global gemini_busy, last_sign, history_log
    gemini_busy = True
    try:
        # Optimization for speed
        pil_img = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
        pil_img.thumbnail((180, 180)) 
        
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=75)
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                "ASL Sign? Reply 1 word/char only. (e.g. A, B, HELLO). If unsure: ?",
                types.Part.from_bytes(data=buf.getvalue(), mime_type='image/jpeg')
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="minimal")
            )
        )

        res_text = response.text.strip().upper().replace(".", "")
        
        if res_text and "?" not in res_text:
            # Logic: If it's a new sign or enough time passed, register it
            history_log.appendleft(f"Detected: {res_text}")
            speak(res_text)
            
            if len(res_text) == 1:
                current_word.append(res_text)
            else:
                if current_word:
                    final_sentence.append("".join(current_word))
                    current_word.clear()
                final_sentence.append(res_text)
            last_sign = res_text
        else:
            history_log.appendleft("AI: Unclear")

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            history_log.appendleft("RATE LIMIT REACHED")
        else:
            history_log.appendleft(f"Error: {error_msg[:15]}")
    finally:
        gemini_busy = False

def draw_dashboard(frame, box, hand_found, progress):
    h, w, _ = frame.shape
    # Glass Sidebar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, "SIGN-LINK 2026", (25, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 180), 2)
    
    # Live Status
    status_color = (0, 165, 255) if gemini_busy else (0, 255, 100)
    status_text = "THINKING..." if gemini_busy else "READY"
    cv2.putText(frame, f"STATUS: {status_text}", (25, 80), 1, 0.9, status_color, 1)

    for i, msg in enumerate(list(history_log)):
        color = (255, 255, 255) if i == 0 else (120, 120, 120)
        cv2.putText(frame, f"> {msg}", (25, 140 + i*35), 1, 0.9, color, 1)

    # Output Bar
    cv2.rectangle(frame, (320, h-85), (w-20, h-15), (15, 15, 15), -1)
    word_text = "".join(current_word) + "_"
    cv2.putText(frame, f"WORD: {word_text}", (340, h-50), 1, 1.5, (0, 210, 255), 2)
    sent_text = " ".join(final_sentence[-3:])
    cv2.putText(frame, f"SENTENCE: {sent_text}", (340, h-25), 1, 0.8, (170, 170, 170), 1)

    if hand_found and box:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 150), 2)
        # Smooth progress bar
        bar_len = int((x2 - x1) * progress)
        cv2.rectangle(frame, (x1, y1-12), (x1 + bar_len, y1-4), (0, 255, 150), -1)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
hand_start_time = None



while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    hand_found = False
    roi_box = None
    progress = 0

    if results.multi_hand_landmarks:
        hand_found = True
        lm = results.multi_hand_landmarks[0]
        xs = [int(p.x * w) for p in lm.landmark]
        ys = [int(p.y * h) for p in lm.landmark]
        
        # Safe Box Calculation
        x1, y1 = max(0, min(xs)-30), max(0, min(ys)-30)
        x2, y2 = min(w, max(xs)+30), min(h, max(ys)+30)
        roi_box = (x1, y1, x2, y2)
        
        if hand_start_time is None: hand_start_time = time.time()
        duration = time.time() - hand_start_time
        progress = min(duration / HOLD_SECONDS, 1.0)

        # TRIGGER AI
        if duration >= HOLD_SECONDS and not gemini_busy and (time.time() - last_call_time) > CAPTURE_INTERVAL:
            # Final safety check on ROI size
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                roi = frame[y1:y2, x1:x2]
                threading.Thread(target=process_sign, args=(roi.copy(),), daemon=True).start()
                last_call_time = time.time()
    else:
        hand_start_time = None

    draw_dashboard(frame, roi_box, hand_found, progress)
    cv2.imshow("Sign-Link AI | Pro Demo", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == 13: # ENTER
        if current_word:
            final_sentence.append("".join(current_word))
            current_word.clear()
    if key == ord('c'):
        current_word, final_sentence = [], []
        history_log.appendleft("Session Reset")

cap.release()
cv2.destroyAllWindows()