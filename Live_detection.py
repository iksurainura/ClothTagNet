import cv2
import numpy as np
import time
import signal
import sys
from ultralytics import YOLO

# ------------------ USER CONFIG ------------------
MODEL_PATH     = "runs/classify/train2/weights/best.pt"
CAMERA_INDEX   = 0
CONF_THRESHOLD = 0.70
WINDOW_SCALE   = 5
VERBOSE        = True
# -------------------------------------------------

PALETTE = {"PASS": (0, 255, 0), "FAIL": (0, 0, 255), "NONE": (160, 160, 160)}

running = True
def signal_handler(sig, frame):
    global running
    running = False
signal.signal(signal.SIGINT, signal_handler)

def Live_feedback():
    global running
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Cannot open camera"); return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    win_name = "Live Fabric Classification"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # choose display size
    ret, frame = cap.read()
    TARGET_W = int(frame.shape[1] * WINDOW_SCALE)
    TARGET_H = int(frame.shape[0] * WINDOW_SCALE)

    fps = 0.0
    fps_counter = 0
    start = time.time()

    while running:
        ret, frame = cap.read()
        if not ret:
            if VERBOSE: print("⚠️  failed to grab frame")
            break

        disp = cv2.resize(frame, (TARGET_W, TARGET_H))

        # ------------- INFERENCE -------------
        results = model(disp, verbose=False)
        probs   = results[0].probs
        class_id = probs.top1
        confidence = float(probs.top1conf)
        class_name = model.names[class_id]

        # decide status
        if class_name.lower() == "good":
            color, status = PALETTE["PASS"], "PASS"
        elif class_name.lower() == "bad":
            color, status = PALETTE["FAIL"], "FAIL"
        else:
            color, status = PALETTE["NONE"], "NONE"

        # ------------- BOX DECISION -------------
        # use YOLO mask ONLY if the model supplied one
        bbox = None
        if hasattr(results[0], 'masks') and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            mask = cv2.resize(mask, (TARGET_W, TARGET_H))
            mask = (mask * 255).astype(np.uint8)
            _, th = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
                bbox = (x, y, w, h)

        # ------------- DRAW -------------
        if bbox and status != "NONE":          # confident + mask exists
            x, y, w, h = bbox
            cv2.rectangle(disp, (x, y), (x + w, y + h), color, 3)
            label = f"{class_name.upper()} {confidence:.2f}"
            cv2.putText(disp, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:                                  # no box – just text
            cv2.putText(disp, status, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # ------------- FPS -------------
        fps_counter += 1
        if fps_counter % 30 == 0:
            end = time.time(); fps = 30/(end-start); start = end
        cv2.putText(disp, f"FPS:{fps:.1f}", (TARGET_W-140, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        cv2.imshow(win_name, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: break
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1: break

    # ---------- CLEANUP ----------
    if VERBOSE: print("🧹 Releasing camera …")
    cap.release(); cv2.destroyAllWindows(); sys.exit(0)

if __name__ == "__main__":
    Live_feedback()