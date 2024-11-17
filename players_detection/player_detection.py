import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("final.pt")

video_path = "./input_video.mp4"
output_path = "./output_video_id_final_model_without_oy.mp4"

# we need to divide frames
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# we need to take into account frames that could be empty - missing detected player
previous_frame = [] 
id_setter = 0 

def calculate_euclidian_distance(this_center_point, previous_center_point):
    """
    Calculating Euclidian distance.
    """
    distances = []

    for this_point in this_center_point:
        point_distances = []
        for prev_point in previous_center_point:
            dist = np.sqrt((this_point[0] - prev_point[0]) ** 2 + (this_point[1] - prev_point[1]) ** 2)
            point_distances.append(dist)
        
        distances.append(point_distances)

    return distances


def tracker(this_frame, previous_frame):
    """
    Setting ids based on the previous frames.
    Sometimes model misses some detections of one player, thus it also creates a box for such cases.
    """
    global id_setter
    refactored_frame = []


    if not previous_frame:
        return [[*box, id_setter + i] for i, box in enumerate(this_frame)], id_setter + len(this_frame) # just need to set ids for the first frame

    this_center_point = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in this_frame]
    previous_center_point = [(int((x1 + x2) / 2), int((y1 + y2) / 2), id_) for x1, y1, x2, y2, id_ in previous_frame]

    distances = calculate_euclidian_distance(this_center_point, previous_center_point)
    
    assigned = set()  
    for i, dist_row in enumerate(distances):
        min_index = np.argmin(dist_row)
        closest_id = previous_center_point[min_index][2] 
        
        if closest_id in [0, 1]:
            refactored_frame.append([*this_frame[i], closest_id])
            assigned.add(min_index) 
        else:
            if id_setter in [0, 1]:
                new_id = id_setter
            else:
                new_id = 0

            refactored_frame.append([*this_frame[i], new_id])

            if new_id == 0:
                id_setter = 1
            else:
                id_setter = 0

    return refactored_frame, id_setter


# def setting_ids(boxes):
#     """
#     Setting ids based on the OY axes
#     """
#     if not boxes:
#         return boxes
    
#     sorted_boxes = sorted(boxes, key=lambda box: box[1])  # Sort by y1 (top of the box)
    
#     remapped_boxes = []
#     for i, box in enumerate(sorted_boxes):
#         x1, y1, x2, y2, id_ = box
#         new_id = 1 if i == 0 else 2  # Highest player gets ID 1, the other gets ID 2
#         remapped_boxes.append([x1, y1, x2, y2, new_id])
    
#     return remapped_boxes

if __name__ == "__main__":
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, save=False, conf=0.3)

        boxes = []
        confidences = []
        for box in results[0].boxes:
            cls_name = results[0].names[int(box.cls[0])]
            if cls_name == "player1": 
                boxes.append(box.xyxy[0].tolist())  # having coords [x1, y1, x2, y2]
                confidences.append(box.conf[0].item())

        if len(boxes) < 2 and previous_frame:
            boxes = [b[:4] for b in previous_frame[:2]] # need to think about how to do it better
        
        this_frame, id_setter = tracker(boxes, previous_frame)

        for box in this_frame:
            x1, y1, x2, y2, id_ = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Player {id_}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        previous_frame = this_frame

        out.write(frame)

    cap.release()
    out.release()

    print(f"Successfuly finished, save to: {output_path}")

    # need to add also not to have just id, but probably to set people's names, etc. -- from command line?


