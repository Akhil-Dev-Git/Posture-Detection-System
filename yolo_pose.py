import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load the YOLOv8 pose model
print("Loading professional pose estimation model...")
model = YOLO('yolov8n-pose.pt') 

# Keypoint connections for YOLO (COCO format - 17 keypoints)
SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),      # Upper body (arms & shoulders)
    (11, 12), (5, 11), (6, 12),                   # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),       # Lower body (legs)
    (0, 1), (0, 2), (1, 3), (2, 4)                # Face
]

# Right joints (even indices after 0) -> Orange
RIGHT_JOINTS = [2, 4, 6, 8, 10, 12, 14, 16]
# Left joints (odd indices) -> Cyan
LEFT_JOINTS = [1, 3, 5, 7, 9, 11, 13, 15]
# Center -> Yellow
CENTER_JOINTS = [0]

def draw_transparent_rect(img, top_left, bottom_right, color, alpha=0.3):
    """Draws a semi-transparent rectangle for a premium HUD effect."""
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully. Press 'q' to quit.")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    
    # Run Inference
    results = model(frame, verbose=False)
    
    # We will create a fresh copy instead of default plot to make it professional
    annotated_frame = frame.copy()
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    persons_detected = 0

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        keypoints = results[0].keypoints.xy.cpu().numpy()
        confs = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else None
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
        box_confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []

        persons_detected = len(keypoints)

        for i in range(persons_detected):
            person_kpts = keypoints[i]
            person_confs = confs[i] if confs is not None else np.ones(17)

            # Draw Professional Bounding Box
            if i < len(boxes):
                x1, y1, x2, y2 = map(int, boxes[i])
                confidence = box_confs[i] if i < len(box_confs) else 1.0
                
                # Corner bracket style bounding box (Professional security camera style)
                line_len = max((x2 - x1) // 6, 10)
                box_color = (0, 200, 100) # Greenish
                
                # Top-Left
                cv2.line(annotated_frame, (x1, y1), (x1+line_len, y1), box_color, 2)
                cv2.line(annotated_frame, (x1, y1), (x1, y1+line_len), box_color, 2)
                # Top-Right
                cv2.line(annotated_frame, (x2, y1), (x2-line_len, y1), box_color, 2)
                cv2.line(annotated_frame, (x2, y1), (x2, y1+line_len), box_color, 2)
                # Bottom-Left
                cv2.line(annotated_frame, (x1, y2), (x1+line_len, y2), box_color, 2)
                cv2.line(annotated_frame, (x1, y2), (x1, y2-line_len), box_color, 2)
                # Bottom-Right
                cv2.line(annotated_frame, (x2, y2), (x2-line_len, y2), box_color, 2)
                cv2.line(annotated_frame, (x2, y2), (x2, y2-line_len), box_color, 2)

                # Label showing confidence
                label = f"Person ID:{i+1} | {confidence*100:.1f}%"
                cv2.rectangle(annotated_frame, (x1, y1-25), (x1 + len(label)*10, y1), box_color, -1)
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw lines (Skeleton) with solid clean white lines
            for (p1, p2) in SKELETON_CONNECTIONS:
                if person_confs[p1] > 0.4 and person_confs[p2] > 0.4:
                    pt1 = (int(person_kpts[p1][0]), int(person_kpts[p1][1]))
                    pt2 = (int(person_kpts[p2][0]), int(person_kpts[p2][1]))
                    
                    if pt1 != (0, 0) and pt2 != (0, 0):
                        # Clean white line mapping
                        cv2.line(annotated_frame, pt1, pt2, (230, 230, 230), 2, cv2.LINE_AA)

            # Draw joints (Keypoints) colored by Left/Right biomechanics
            for j in range(17):
                if person_confs[j] > 0.4:
                    pt = (int(person_kpts[j][0]), int(person_kpts[j][1]))
                    if pt != (0, 0):
                        color = (0, 255, 255) # Default yellow
                        if j in RIGHT_JOINTS:
                            color = (0, 100, 255) # Orange for Right side
                        elif j in LEFT_JOINTS:
                            color = (255, 255, 0) # Cyan for Left side
                        
                        # Sleek circle design mapping
                        cv2.circle(annotated_frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA) # White core
                        cv2.circle(annotated_frame, pt, 5, color, 1, cv2.LINE_AA)             # Outer ring

    # HUD Overlay (Professional Display)
    draw_transparent_rect(annotated_frame, (0, 0), (width, 50), (10, 10, 10), alpha=0.6)
    
    cv2.putText(annotated_frame, "AI VISION : POSITIONAL ANALYSIS", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Status and metrics
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (width - 120, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2, cv2.LINE_AA)
    
    cv2.putText(annotated_frame, f"TARGETS: {persons_detected}", (width - 280, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

    # Show result
    cv2.imshow("Professional Architecture", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
