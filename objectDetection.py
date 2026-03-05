import cv2
import time
import pyttsx3
from ultralytics import YOLO

# --------- SETUP ---------
model = YOLO("yolov8n.pt")
engine = pyttsx3.init()
engine.setProperty("rate", 160)
CONF = 0.45
CAM = 0
SPEAK_GAP = 1.2  # seconds

last_spoken = 0
current_visible = set()  # objects already announced in current frame set

# --------- SPEAK FUNCTION ---------
def speak(t):
    engine.say(t)
    engine.runAndWait()

# --------- MAIN ---------
cap = cv2.VideoCapture(CAM)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF, verbose=False)
    new_visible = set()

    for box in results[0].boxes:
        cls = int(box.cls)
        label = model.names[cls]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        conf = float(box.conf[0])

        # Draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Track this object in this frame
        new_visible.add(label)

        # ---------- ANNOUNCEMENT LOGIC ----------
        # Speak only if:
        # 1. This is a NEW object not in previous frame
        # 2. Cooldown gap satisfied
        now = time.time()
        if label not in current_visible and (now - last_spoken) > SPEAK_GAP:
            speak(label)
            last_spoken = now

    # Update visible set for next frame
    current_visible = new_visible

    cv2.imshow("Object Detection for Blind", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
