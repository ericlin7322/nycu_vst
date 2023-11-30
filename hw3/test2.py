import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from scipy.optimize import linear_sum_assignment
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained detection model and move it to GPU
model = fasterrcnn_resnet50_fpn(pretrained=True)
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

# Hungarian algorithm for assignment
def hungarian_algorithm(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))

# Function to update object tracks based on assignments
def update_object_tracks(assignment, person_boxes):
    for track_idx, detection_idx in assignment:
        object_tracks[track_idx] = {'box': person_boxes[detection_idx], 'id':track_idx}

# Video capture
cap = cv2.VideoCapture('easy_9.mp4')

used = {}


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
    threshold = 0.7
    person_boxes = boxes[scores > threshold]


    if not object_tracks:
        # Initialize object_tracks on the first frame
        for box in person_boxes:
            object_tracks[track_id_counter] = {'box': box, 'id': track_id_counter}
            track_id_counter += 1
    else:
        # for u in used:
        #     person_boxes = np.insert(person_boxes, u, object_tracks[u]['box'])

        cost_matrix = np.zeros((len(object_tracks), len(person_boxes)))
        for i, prev_box in enumerate(object_tracks.values()):
            for j, current_box in enumerate(person_boxes):
                cost_matrix[i, j] = np.linalg.norm(prev_box['box'] - current_box)

        # Use Hungarian algorithm to assign detections to tracks

        assignment = hungarian_algorithm(cost_matrix)


        new_assignment = []

        # Update object tracks based on assignment
        update_object_tracks(assignment, person_boxes)
        print(object_tracks.keys())

        # for a in assignment:
        # #     # print(a)
        #     x,y = a
        #     for u in used:
        #         if x >= u:
        #             x+=1
        #     new_assignment.append((x,y))

        tid = 0

        for u in used:
            for a in assignment:
                x,y = a
                if x == u:
                    assignment[u] = (max(assignment, key=lambda x: x[0])[0]+tid, y)
                    tid+=1

        print(f"new assignment {assignment}")

        unmatched_tracks = set(range(len(object_tracks))) - set(np.array(assignment)[:, 0])
        for track_idx in unmatched_tracks:
            # print(object_tracks.keys(), np.array(assignment)[:, 0])
            used[track_idx] = object_tracks[track_idx]
            print(track_idx)
            # object_tracks[track_idx] = {'box': person_boxes[track_idx], 'id': track_id_counter}
            # del object_tracks[track_idx]
            # object_tracks[track_id_counter] = {'box': [0,0,0,0], 'id':track_id_counter}
            # track_id_counter += 1
            # mp -= 1

        # unmatched_tracks =  set(np.array(new_assignment)[:, 1]) - set(range(len(object_tracks)))
        # for track_idx in unmatched_tracks:
        #     # print(object_tracks.keys(), np.array(assignment)[:, 0])
        #     # object_tracks[track_idx] = {'box': person_boxes[track_idx], 'id': track_id_counter}
        #     # del object_tracks[track_idx]
        #     object_tracks[track_id_counter] = {'box': person_boxes[track_idx], 'id':track_id_counter}
        #     track_id_counter += 1

        unmatched_tracks =  set(np.array(assignment)[:, 0]) - set(range(len(object_tracks)))
        for track_idx in unmatched_tracks:
            # print("HERE")
            # print(track_idx)
        #     # print(object_tracks.keys(), np.array(assignment)[:, 0])
        #     # del object_tracks[track_idx]
            temp = np.array(assignment)[:, 0]
            print(temp)
            object_tracks[track_id_counter] = {'box': person_boxes[np.where(temp==track_idx)[0][0]], 'id':track_id_counter}
            track_id_counter += 1


        # unmatched_tracks = set(range(len(object_tracks))) - set(np.array(new_assignment)[:, 0])
        # for track_idx in unmatched_tracks:
        #     # print(object_tracks.keys(), np.array(assignment)[:, 0])
        #     used.append(track_idx)
        

    # Draw bounding boxes and count total number of people
    total_people = track_id_counter
    for track_id, data in object_tracks.items():
        color = (0, 255, 0)  # You can choose a fixed color for each person if needed
        box = data['box']
        id = data['id']
        frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        frame = cv2.putText(frame, f'Track {id}', (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    frame = cv2.putText(frame, f'Total People: {total_people}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Object Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
