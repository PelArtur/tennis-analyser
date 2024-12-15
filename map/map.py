import cv2
import numpy as np
import sys
sys.path.append('../')

from torchvision import models, transforms
import cv2 as cv
import numpy as np
import torch
from ultralytics import YOLO

from collections import deque
import pandas


SINGLE_LINE_WIDTH = 8.23
DOUBLE_LINE_WIDTH = 10.97
HALF_COURT_LINE_HEIGHT = 11.88 #height of the court / 2
SERVICE_LINE_WIDTH = 6.4
DOUBLE_ALLY_DIFFERENCE = 1.37
NO_MANS_LAND_HEIGHT = 5.48 #width of the court / 2



PLAYER_1_HEIGHT_METERS = 1.88
PLAYER_2_HEIGHT_METERS = 1.91


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def load_model(model_path):
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 14*2) 
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_keypoints(model, image, transform):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_rgb).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    keypoints = outputs.squeeze().cpu().numpy()
    original_h, original_w = image.shape[:2]
    keypoints[::2] *= original_w / 224.0
    keypoints[1::2] *= original_h / 224.0

    return keypoints

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)
def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def getposition(bbox):
    x1, y1, x2, y2 = bbox
    return (x2, y2)

def get_height_of_bbox(bbox):
    return bbox[3]-bbox[1]

def measure_xy_distance(p1,p2):
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])

def get_center_of_bbox(bbox):
    return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
   closest_distance = float('inf')
   key_point_ind = keypoint_indices[0]
   for keypoint_indix in keypoint_indices:
       keypoint = keypoints[keypoint_indix*2], keypoints[keypoint_indix*2+1]
       distance = abs((point[1] - keypoint[1]))


       if distance<closest_distance:
           closest_distance = distance
           key_point_ind = keypoint_indix
    
   return key_point_ind


def convert_pixel_distance_to_meters(pixel_distance, refrence_height_in_meters, refrence_height_in_pixels):
    return (pixel_distance * refrence_height_in_meters) / refrence_height_in_pixels

def convert_meters_to_pixel_distance(meters, refrence_height_in_meters, refrence_height_in_pixels):
    return (meters * refrence_height_in_pixels) / refrence_height_in_meters

def get_height_of_bbox(bbox):
    return bbox[3]-bbox[1]

def measure_xy_distance(p1,p2):
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])

def get_center_of_bbox(bbox):
    return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))


def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

class MiniCourt():
    def __init__(self,frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court=20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def draw_bounces(self, frame):
        print(self.bounce_points)
        for point in self.bounce_points:
            cv2.circle(
            frame,            
            (int(point[0]), int(point[1])), 
            10,                 
            (0, 0, 255),       
            -1                 
        )


    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                            )

    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(HALF_COURT_LINE_HEIGHT*2)
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_to_pixels(DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(NO_MANS_LAND_HEIGHT)
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(NO_MANS_LAND_HEIGHT)
        drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_to_pixels(SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 
        self.drawing_key_points=drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (5,7),
            (12,13),
            (2, 3)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self,frame):
        frame= frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self,frame):
        for i in range(0, len(self.drawing_key_points),2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x,y),3, (0,0,255),-1)

        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self,frame):
        shapes = np.zeros_like(frame,np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha=0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def get_mini_court_coordinates(self,
                                object_position,
                                closest_key_point, 
                                closest_key_point_index, 
                                player_height_in_pixels,
                                player_height_in_meters):
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels,
                                                                        player_height_in_meters,
                                                                        player_height_in_pixels)
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels,
                                                                            player_height_in_meters,
                                                                            player_height_in_pixels)
        
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)

        closest_mini_court_keypoint = (self.drawing_key_points[closest_key_point_index * 2],
                                    self.drawing_key_points[closest_key_point_index * 2 + 1])

        mini_court_player_position = (closest_mini_court_keypoint[0] + mini_court_x_distance_pixels,
                                    closest_mini_court_keypoint[1] + mini_court_y_distance_pixels)

        x = min(max(self.court_start_x, mini_court_player_position[0]), self.court_end_x - 1)
        y = min(max(self.court_start_y, mini_court_player_position[1]), self.court_end_y - 1)

        return (x, y)

    def convert_ball_coordinates_to_mini_court(self, frame_num, ball_positions, original_court_key_points):
        output_ball_positions = {}
        
        closest_key_point_index = get_closest_keypoint_index(
            ball_positions, original_court_key_points, [0, 2, 12, 13]
        )
        closest_key_point = (
            original_court_key_points[closest_key_point_index * 2],
            original_court_key_points[closest_key_point_index * 2 + 1]
        )

        mini_court_ball_position = self.get_mini_court_coordinates(
            ball_positions,
            closest_key_point,
            closest_key_point_index,
            5, 
            0.067 
        )

        output_ball_positions[frame_num] = mini_court_ball_position
        self.bounce_points.append(output_ball_positions[frame_num])
        return output_ball_positions[frame_num]

    
    
    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, original_court_key_points):
        player_heights = {
            0: PLAYER_1_HEIGHT_METERS,
            1: PLAYER_2_HEIGHT_METERS
        }

        output_player_positions = {}

        for frame_num, player_bbox in enumerate(player_boxes):
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)
                print(player_id)
                frame_index_min = max(0, frame_num-20)
                frame_index_max = min(len(player_boxes), frame_num+50)
                bboxes_heights_in_pixels = [get_height_of_bbox(player_boxes[i][player_id]) for i in range (frame_index_min,frame_index_max)]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                closest_key_point_index = get_closest_keypoint_index(foot_position, original_court_key_points, [0, 2, 13])
                closest_key_point = (
                    original_court_key_points[closest_key_point_index * 2],
                    original_court_key_points[closest_key_point_index * 2 + 1]
                )
                if player_id == 0:
                    max_player_height_in_pixels = 62
                    foot_position = getposition(bbox)
                    player_heights[0] = 1.60

                    

                
                mini_court_player_position = self.get_mini_court_coordinates(
                    foot_position,
                    closest_key_point,
                    closest_key_point_index,
                    max_player_height_in_pixels,
                    player_heights[player_id]
                )

                output_player_positions[player_id] = mini_court_player_position

        return output_player_positions
    
    def draw_players(self, player_positions):
        court_with_players = self.court.copy()
        for player_id, (x, y) in player_positions.items():
            x = min(max(0, int(x)), self.court_width - 1)
            y = min(max(0, int(y)), self.court_height - 1)

            cv2.circle(court_with_players, (x, y), 5, self.circle_color, -1)
            cv2.putText(court_with_players, str(player_id), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.line_color, 1)

        return court_with_players
    


def match_mini_court_keypoints(mini_court, predicted_keypoints):
    mini_court_points = mini_court.get_court_drawing_keypoints()
    mini_court_points = np.array(mini_court_points).reshape(-1, 2)  
    predicted_points = np.array(predicted_keypoints).reshape(-1, 2)
    
    matches = {}
    for i, mini_point in enumerate(mini_court_points):
        closest_index = np.argmin([np.linalg.norm(mini_point - pred_point) for pred_point in predicted_points])
        matches[i + 1] = closest_index + 1  
    return matches

def tracker(this_frame, previous_frame):
    global id_setter
    refactored_frame = []


    if not previous_frame:
        return [[*box, id_setter + i] for i, box in enumerate(this_frame)], id_setter + len(this_frame) 

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

def calculate_euclidian_distance(this_center_point, previous_center_point):
    distances = []

    for this_point in this_center_point:
        point_distances = []
        for prev_point in previous_center_point:
            dist = np.sqrt((this_point[0] - prev_point[0]) ** 2 + (this_point[1] - prev_point[1]) ** 2)
            point_distances.append(dist)
        
        distances.append(point_distances)

    return distances


def calculate_speed(prev_pos, curr_pos, fps):
    if prev_pos is None:
        return 0
    distance = ((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2) ** 0.5
    speed = distance * fps  

    coefficient = HALF_COURT_LINE_HEIGHT / 500 # need to think about * 2
    speed = speed * coefficient

    return speed * 3.6


def project_to_minimap(x_3d, y_3d, field_bounds, map_size):

    (x_min, x_max), (y_min, y_max) = field_bounds
    map_width, map_height = map_size

    x_2d = (x_3d - x_min) / (x_max - x_min) * map_width
    y_2d = (y_3d - y_min) / (y_max - y_min) * map_height


    return x_2d, y_2d


model1 = YOLO("final.pt")
model_path = "model_epoch_18.pth"
model = load_model(model_path)
transform = get_transform()

video_path = "input_video.mp4"
output_video_path = "output_video_final.avi"

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = int(cap.get(cv2.CAP_PROP_FPS))

ball_bounces = pandas.read_csv("ball_coords.csv")
bounce_coordinates = ball_bounces[ball_bounces['Frames'].notna()]

bounce_dict = {row['Frames']: (row['X'], row['Y']) for _, row in bounce_coordinates.iterrows()}

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

mini_court = None
previous_frame = []
previous_positions = {0: None, 1: None}
speed_buffer = {0: deque(maxlen=4), 1: deque(maxlen=4)}
frame_counter = 0
id_setter = 0
smoothed_speed1, smoothed_speed2 = 0, 0
speed_box_x, speed_box_y = 10, 10
speed_box_width, speed_box_height = 200, 60 
alpha = 0.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    if mini_court is None:
        mini_court = MiniCourt(frame)

    results = model1.predict(source=frame, save=False, conf=0.3)

    boxes = []
    confidences = []
    for box in results[0].boxes:
        cls_name = results[0].names[int(box.cls[0])]
        if cls_name == "player1":
            boxes.append(box.xyxy[0].tolist())  
            confidences.append(box.conf[0].item())

    if len(boxes) < 2 and previous_frame:
        boxes = [b[:4] for b in previous_frame[:2]] 

    this_frame, id_setter = tracker(boxes, previous_frame)

    for box in this_frame:
        x1, y1, x2, y2, id_ = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {id_}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    frame_with_background = mini_court.draw_background_rectangle(frame)
    keypoints = predict_keypoints(model, frame, transform)
 

    image_with_keypoints = frame_with_background.copy()

    for idx, (x, y) in enumerate(keypoints.reshape(-1, 2)):
        x, y = int(x), int(y)
        cv2.circle(image_with_keypoints, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(image_with_keypoints, str(idx + 1), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    mini_court_points = np.array(mini_court.get_court_drawing_keypoints()).reshape(-1, 2)
    for idx, (x, y) in enumerate(mini_court_points):
        x, y = int(x), int(y)
        cv2.circle(image_with_keypoints, (x, y), 2, (255, 0, 0), -1)
        cv2.putText(image_with_keypoints, str(idx + 1), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    keypoint_matches = match_mini_court_keypoints(mini_court, keypoints)
    for mini_idx, pred_idx in keypoint_matches.items():
        mini_point = tuple(map(int, mini_court_points[mini_idx - 1]))
        pred_point = tuple(map(int, keypoints[(pred_idx - 1) * 2:(pred_idx - 1) * 2 + 2]))

    frame_with_mini_court = mini_court.draw_court(image_with_keypoints)
    player_boxes = {int(box[4]): box[:4] for box in this_frame}  
    player_positions = mini_court.convert_bounding_boxes_to_mini_court_coordinates([player_boxes], keypoints)
    cv2.circle(frame_with_mini_court, (int(player_positions[0][0]), int(player_positions[0][1])), 5, (0, 255, 255), -1)
    cv2.circle(frame_with_mini_court, (int(player_positions[1][0]), int(player_positions[1][1])), 5, (0, 255, 255), -1)


    ball_coords = bounce_dict[frame_counter]
    # positions = mini_court.convert_ball_coordinates_to_mini_court(frame_counter, ball_coords, keypoints)

    new_coords = project_to_minimap(ball_coords[0], ball_coords[1], ((199.8, 1076), (207.2, 532.76)), (250, 500))
    cv2.circle(frame_with_mini_court, (int(new_coords[0] + mini_court.start_x), int(new_coords[1]+mini_court.start_y)), 2, (0, 0, 255), -1)
       

    mini_court.draw_bounces(image_with_keypoints)

    speed1 = calculate_speed(previous_positions[0], player_positions[0], fps)
    speed2 = calculate_speed(previous_positions[1], player_positions[1], fps)

    speed_buffer[0].append(speed1)
    speed_buffer[1].append(speed2)

    if frame_counter % 4 == 0:
        smoothed_speed1 = sum(speed_buffer[0]) / len(speed_buffer[0]) if len(speed_buffer[0]) > 0 else 0
        smoothed_speed2 = sum(speed_buffer[1]) / len(speed_buffer[1]) if len(speed_buffer[1]) > 0 else 0
        
   
    overlay = image_with_keypoints.copy()
    
    cv2.rectangle(overlay, (speed_box_x, speed_box_y), (speed_box_x + speed_box_width, speed_box_y + speed_box_height), (255, 255, 255), -1)
    cv2.addWeighted(overlay, alpha, image_with_keypoints, 1 - alpha, 0, image_with_keypoints)

    cv2.putText(image_with_keypoints, f"Player 0: {smoothed_speed1:.2f} km/h", (speed_box_x + 10, speed_box_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.putText(image_with_keypoints, f"Player 1: {smoothed_speed2:.2f} km/h", (speed_box_x + 10, speed_box_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    out.write(frame_with_mini_court)

    previous_frame = this_frame

    previous_positions[0] = player_positions[0]
    previous_positions[1] = player_positions[1]

cap.release()
out.release()


