import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Load pre-trained Faster R-CNN model with a person-only class
def load_pretrained_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)  # 2 classes (background + person)
    return model

# Apply the model to detect persons in a frame
def detect_persons(model, frame, threshold=0.5):
    transform = T.Compose([T.ToTensor()])
    img = transform(frame)
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        model.eval()
        prediction = model(img)
    
    boxes = prediction[0]['boxes']
    scores = prediction[0]['scores']
    labels = prediction[0]['labels']
    
    persons = [boxes[i] for i in range(len(boxes)) if scores[i] > threshold and labels[i] == 1]  # Person class is assumed to be 1
    
    return persons

# Apply Hungarian algorithm to assign IDs to tracked objects
def hungarian_algorithm(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))

# Simple centroid tracking for demonstration purposes
class CentroidTracker:
    def __init__(self, max_disappeared=5):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, centroids):
        if not centroids:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.objects

        input_centroids = np.array(centroids)

        if not self.objects:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            cost_matrix = np.linalg.norm(object_centroids - input_centroids, axis=1)
            rows, cols = hungarian_algorithm(cost_matrix)

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(len(object_centroids))) - used_rows
            unused_cols = set(range(len(input_centroids))) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

# Main function for object tracking
def object_tracking(model, video_path):
    cap = cv2.VideoCapture(video_path)
    centroid_tracker = CentroidTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        persons = detect_persons(model, frame)

        centroids = []
        for person in persons:
            x, y, w, h = map(int, person)
            centroid = (x + w // 2, y + h // 2)
            centroids.append(centroid)

        tracked_objects = centroid_tracker.update(centroids)

        for object_id, centroid in tracked_objects.items():
            cv2.putText(frame, f"ID {object_id}", (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, centroid, 4, (0, 255, 0), -1)

        cv2.imshow('Object Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load pre-trained Faster R-CNN model
    model = load_pretrained_model()

    # Specify the path to the video file
    video_path = 'easy_9.mp4'

    # Perform object tracking
    object_tracking(model, video_path)
