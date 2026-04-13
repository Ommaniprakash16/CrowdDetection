import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import OrderedDict
from scipy.spatial import distance as dist

# =====================================================
# ---------------- Centroid Tracker -------------------
# =====================================================
class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, centroids):
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for c in centroids:
                self.register(c)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = dist.cdist(np.array(object_centroids), np.array(centroids))
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            if D[r, c] > self.max_distance:
                continue

            object_id = object_ids[r]
            self.objects[object_id] = centroids[c]
            self.disappeared[object_id] = 0

            used_rows.add(r)
            used_cols.add(c)

        for r in range(D.shape[0]):
            if r not in used_rows:
                object_id = object_ids[r]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

        for c in range(len(centroids)):
            if c not in used_cols:
                self.register(centroids[c])

        return self.objects


# =====================================================
# ---------------- CONFIG ------------------------------
# =====================================================
MODEL_PATH = "models\yolov8s.pt"
VIDEO_PATH = r"Input\block2video.mp4"

 # <-- PUT YOUR VIDEO HERE
CONF_THRESHOLD = 0.35
RESIZE_WIDTH = 640

SKIP_FRAMES = 2

ALERT_THRESHOLD = 30
DENSITY_THRESHOLD = 0.27

# =====================================================
# ---------------- LOAD MODEL --------------------------
# =====================================================
model = YOLO(MODEL_PATH)
model.overrides = {"conf": CONF_THRESHOLD}

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Cannot open video")
    exit()

ret, frame = cap.read()
h0, w0 = frame.shape[:2]

scale = RESIZE_WIDTH / w0
RESIZE_DIM = (RESIZE_WIDTH, int(h0 * scale))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# Polygon zone (10% margins)
w, h = RESIZE_DIM
margin_x = int(w * 0.10)
margin_y = int(h * 0.10)

polygon = np.array([
    [margin_x, margin_y],
    [w - margin_x, margin_y],
    [w - margin_x, h - margin_y],
    [margin_x, h - margin_y]
])

zone = sv.PolygonZone(polygon=polygon)
zone_area = float(abs(cv2.contourArea(polygon)))

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

tracker = CentroidTracker()
unique_ids = set()

# =====================================================
# ---------------- MAIN LOOP ---------------------------
# =====================================================
frame_count = 0
t0 = time.time()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % (SKIP_FRAMES + 1) != 0:
        continue

    frame = cv2.resize(frame, RESIZE_DIM)

    # ---------------- YOLO detection ----------------
    results = model.predict(frame, classes=[0], verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])

    mask = detections.confidence >= CONF_THRESHOLD
    detections = detections[mask]

    # ---------------- Centroids ----------------
    centroids = []
    for bb in detections.xyxy:
        x1, y1, x2, y2 = map(int, bb)
        centroids.append(((x1+x2)//2, (y1+y2)//2))

    objects = tracker.update(centroids)

    # ---------------- Zone filter ----------------
    inside_mask = zone.trigger(detections=detections)
    detections_zone = detections[inside_mask]

    ids_in_zone = set()

    for bb in detections_zone.xyxy:
        cx = int((bb[0]+bb[2])/2)
        cy = int((bb[1]+bb[3])/2)

        min_dist = 1e9
        match = None

        for oid, cent in objects.items():
            d = (cent[0]-cx)**2 + (cent[1]-cy)**2
            if d < min_dist:
                min_dist = d
                match = oid

        if match is not None:
            ids_in_zone.add(match)
            unique_ids.add(match)

    current_count = len(ids_in_zone)
    unique_count = len(unique_ids)

    # ---------------- Density ----------------
    total_area = 0
    for bb in detections_zone.xyxy:
        x1,y1,x2,y2 = map(int, bb)
        total_area += (x2-x1)*(y2-y1)

    density = total_area / zone_area

    # ---------------- Drawing ----------------
    frame = box_annotator.annotate(frame, detections_zone)

    for oid, cent in objects.items():
        cv2.circle(frame, cent, 4, (255,0,0), -1)
        cv2.putText(frame, f"ID:{oid}", cent, 0, 0.5, (255,0,0), 1)

    cv2.polylines(frame, [polygon], True, (0,255,0), 2)

    fps = frame_count / (time.time()-t0)

    cv2.putText(frame, f"Count: {current_count}", (10,30),
                0, 1, (0,255,0), 2)
    cv2.putText(frame, f"Unique: {unique_count}", (10,60),
                0, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"Density: {density:.2f}", (10,90),
                0, 0.7, (255,255,0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10,120),
                0, 0.7, (255,255,255), 2)

    # ---------------- Alerts ----------------
    if current_count >= ALERT_THRESHOLD:
        cv2.putText(frame, "CROWD ALERT!", (300,40),
                    0, 1, (0,0,255), 3)

    if density >= DENSITY_THRESHOLD:
        cv2.putText(frame, "HIGH DENSITY!", (300,80),
                    0, 1, (0,0,255), 3)

    # ---------------- Show ----------------
    cv2.namedWindow("Crowd Monitoring", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Crowd Monitoring", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Crowd Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
