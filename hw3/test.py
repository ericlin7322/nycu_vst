import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from scipy.optimize import linear_sum_assignment
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained detection model and move it to GPU
model = keypointrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# Define transformation for input images
transform = T.Compose([T.ToTensor()])

# Initialize variables for object tracking
object_tracks = {}
track_id_counter = 0

# Number of frames to wait before considering a track as inactive
inactive_threshold = 10
inactive_tracks = {}

def shrink_box(box):
    # Calculate the center coordinates
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2

    # Calculate the new half-width and half-height
    half_width = (box[2] - box[0]) / 4  # Shrink width by half
    half_height = (box[3] - box[1]) / 4  # Shrink height by half

    # Calculate new coordinates of the bounding box
    new_box = (
        center_x - half_width,  # x_min
        center_y - half_height,  # y_min
        center_x + half_width,  # x_max
        center_y + half_height   # y_max
    )

    return new_box

def is_box_within_frame(box, frame_width, frame_height, offset):
    x1, y1, x2, y2 = box
    centerx, centery = x1 + (x2-x1) / 2, y1 + (y2-y1) / 2
    # print(centerx, centery)

    # Check if all corners of the bounding box are within the frame
    if offset <= x1 < frame_width - offset and offset <= y1 <= frame_height-offset and offset <= x2 <= frame_width-offset and offset <= y2 <= frame_height-offset:
    # if offset <= centerx < frame_width - offset and offset <= centery <= frame_height-offset:
        return True
    else:
        return False

def calculate_distance(box1, box2):
    center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
    center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])

    # Calculate the Euclidean distance between the centers
    distance = np.linalg.norm(center1 - center2)

    return distance

def calculate_iou(box1, box2):
    # Calculate the intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the Union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

# Hungarian algorithm for assignment
def hungarian_algorithm(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))

# Function to update object tracks based on assignments
def update_object_tracks(assignment, person_boxes):
    for track_idx, detection_idx in assignment:
        object_tracks[track_idx] = {'box': person_boxes[detection_idx], 'id':track_idx, 'isInBox': object_tracks[track_idx]['isInBox']}
        if is_box_within_frame(object_tracks[track_idx]['box'], 1280, 720, 30):
            object_tracks[track_idx]['isInBox'] = True


# Video capture
cap = cv2.VideoCapture('easy_9.mp4')

last_assignment = [(0, 0)]
last_person_length = -1
last_person_box = []
used_box = []
ind = []
person_count = 0

colors_array = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),   # Maroon
    (0, 128, 0),   # Olive
    (0, 0, 128),   # Navy
    (128, 128, 0),  # Olive Green
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (128, 128, 128),  # Gray
    (192, 192, 192),  # Silver
    (255, 165, 0),  # Orange
    (128, 0, 0),   # Brown
    (165, 42, 42),  # Brown
    (0, 128, 128),  # Dark Teal
    (255, 140, 0),  # Dark Orange
    (0, 128, 0),   # Dark Green
    (255, 0, 127),  # Pink
    (139, 69, 19),  # Saddle Brown
    (0, 0, 139),   # Dark Blue
    (255, 215, 0),  # Gold
    (70, 130, 180),  # Steel Blue
    (255, 69, 0),   # Red-Orange
    (218, 112, 214),  # Orchid
    (173, 255, 47),  # Green Yellow
    (240, 230, 140),  # Khaki
    (255, 20, 147),  # Deep Pink
    (70, 130, 180),  # Steel Blue
    (255, 69, 0),   # Red-Orange
    (218, 112, 214),  # Orchid
    (173, 255, 47),  # Green Yellow
    (240, 230, 140),  # Khaki
    (255, 20, 147),  # Deep Pink
    (0, 128, 0),   # Dark Green
    (255, 0, 127),  # Pink
    (139, 69, 19),  # Saddle Brown
    (0, 0, 139),   # Dark Blue
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the current frame
    with torch.no_grad():
        img_tensor = transform(frame).unsqueeze(0).to(device)
        detections = model(img_tensor)
    
    person_indices = (detections[0]['labels'] == 1).nonzero().squeeze(1)

    # Convert detection results to numpy arrays
    boxes = detections[0]['boxes'][person_indices].cpu().numpy()
    scores = detections[0]['scores'][person_indices].cpu().numpy()

    # Threshold for considering a detection as a person
    threshold = 0.95
    person_boxes = boxes[scores > threshold]

    if not object_tracks:
        for box in person_boxes:
            object_tracks[track_id_counter] = {'box': box, 'id': track_id_counter, 'isInBox': True}
            ind.append(track_id_counter)
            track_id_counter += 1
            
    else:
        if last_person_length != -1:
            if last_person_length < len(person_boxes):
                    # print("HERE")
                    for track_idx in range(len(person_boxes)):
                        if is_box_within_frame(person_boxes[track_idx], 1280, 720, 10) == False:
                            isSame = False
                            for j in object_tracks.values():
                                # print(calculate_iou(person_boxes[track_idx], j['box']))
                                if calculate_iou(person_boxes[track_idx], j['box'])>0.2:
                                    isSame = True
                            if isSame == False:
                                object_tracks[track_id_counter] = {'box': person_boxes[track_idx], 'id':track_id_counter, 'isInBox': False}
                                # ind[track_idx] = track_id_counter
                                track_id_counter+=1
                                # print(i, calculate_iou(object_tracks[track_idx]['box'], person_boxes[track_idx]))
                            # print()
            elif last_person_length > len(person_boxes):
                for track_idx in range(len(last_person_box)):
                        if is_box_within_frame(last_person_box[track_idx], 1280, 720, 10) == False:
                            # isSame = False
                            for i, j in enumerate(object_tracks.values()):
                            #     # print(calculate_iou(person_boxes[track_idx], j['box']))
                                if calculate_iou(last_person_box[track_idx], j['box']) > 0.9:
                                    new_box = (object_tracks[i]['box'][0],object_tracks[i]['box'][1],object_tracks[i]['box'][0],object_tracks[i]['box'][1])
                                    # print(new_box)
                                    object_tracks[i]['box'] =  (object_tracks[i]['box'][0],object_tracks[i]['box'][1],object_tracks[i]['box'][0],object_tracks[i]['box'][1])
                                    # person_count+=1
                            #         isSame = True
                            #     if isSame == False:
                            # used_box.append(track_idx)
                            # del object_tracks[track_idx]
                            # ind[track_idx] = -1
                            # new_box = (object_tracks[track_idx]['box'][0],object_tracks[track_idx]['box'][1],object_tracks[track_idx]['box'][0],object_tracks[track_idx]['box'][1])
                            # print(new_box)
                            # object_tracks[track_idx]['box'] =  (object_tracks[track_idx]['box'][0],object_tracks[track_idx]['box'][1],object_tracks[track_idx]['box'][0],object_tracks[track_idx]['box'][1])
                            # pass

        cost_matrix = np.zeros((len(object_tracks), len(person_boxes)))
        for i, prev_box in enumerate(object_tracks.values()):
            for j, current_box in enumerate(person_boxes):
                box1 = prev_box['box']
                box2 = current_box
                center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
                center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
                cost_matrix[i, j] = np.linalg.norm(prev_box['box'] - current_box)
                # print(cost_matrix[i,j])
                # cost_matrix[i, j] = center1 - center2

        assignment = hungarian_algorithm(cost_matrix) 

        # for u in used_box:
        #     for i, a in enumerate(assignment):
        #         x,y = a
        #         if x >= u:
        #             assignment[i] = x+1, y

        update_object_tracks(assignment, person_boxes)
        # print(object_tracks.keys())
        # print(f"new assignment {assignment}")

        last_assignment = assignment
        last_person_length = len(person_boxes)
        last_person_box = person_boxes
        
    # Draw bounding boxes and count total number of people
    total_people = track_id_counter
    for track_id, data in object_tracks.items():
        color = colors_array[track_id]
        # color = tuple(map(int, color))  # Ensure that color values are integers
        box = data['box']
        id = data['id']
        if is_box_within_frame(box, 1280,720, 0) and box[0] != box[3]:
            frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            # frame = cv2.putText(frame, f'Track {id}', (int(box[0]), int(box[1]) - 10),
                            # cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # frame = cv2.putText(frame, f'Total People: {total_people}', (10, 30),
                        # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Object Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for a in object_tracks.values():
    if a['isInBox']:
        person_count+=1
print("Totol people: ", person_count)
cap.release()
cv2.destroyAllWindows()
