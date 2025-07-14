import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load YOLO model
model = YOLO("/home/kamal/Desktop/Stealth Mode /Task-2/best.pt")

# Webcam input
cap = cv2.VideoCapture(0)

# Tracking parameters
track_db = {}
next_id = 1
frame_num = 0
circle_radius = 25  # for drawing
fps_target = 30
fps_interval = 1

# Function to get dominant jersey color in HSV
def get_dominant_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    return int(np.argmax(hist))

# FPS calculation
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    # Inference
    results = model(frame)[0]
    detections = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []

    centroids_drawn = []
    current_ids = []

    for box in detections:
        x1, y1, x2, y2 = map(int, box[:4])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        h = y2 - y1

        # Crop region of player
        player_crop = frame[y1:y2, x1:x2]
        if player_crop.size == 0:
            continue

        jersey_color = get_dominant_color(player_crop)

        # Track matching
        matched_id = None
        min_dist = 50
        color_thresh = 15
        height_thresh = 0.2

        for tid, tdata in track_db.items():
            px, py = tdata["centroid"]
            ph = tdata["height"]
            pd_color = tdata["color"]

            dist = np.linalg.norm([cx - px, cy - py])
            color_diff = abs(jersey_color - pd_color)
            height_diff = abs(h - ph) / max(h, ph)

            if dist < min_dist and color_diff <= color_thresh and height_diff < height_thresh:
                matched_id = tid
                break

        # New ID if no match
        if matched_id is None:
            matched_id = next_id
            next_id += 1

        # Update tracking info
        track_db[matched_id] = {
            "centroid": (cx, cy),
            "color": jersey_color,
            "height": h,
            "last_frame": frame_num
        }
        current_ids.append(matched_id)

        # Draw tracking info
        cv2.circle(frame, (cx, cy), circle_radius, (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {matched_id}", (cx - 10, cy - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        centroids_drawn.append((cx, cy))

    # Show frame
    cv2.imshow("YOLO Tracking", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
