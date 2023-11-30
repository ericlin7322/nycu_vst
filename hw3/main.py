import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from scipy.optimize import linear_sum_assignment
import numpy as np

def generate_colors(num_colors):
    # Generate a list of distinct colors
    return [(int(np.random.rand() * 255), int(np.random.rand() * 255), int(np.random.rand() * 255)) for _ in range(num_colors)]

def get_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def detect_people(frame, model, device):
    transform = T.Compose([T.ToTensor()])
    img = transform(frame).to(device)
    img = img.unsqueeze(0)

    with torch.no_grad():
        predictions = model(img)

    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    # Filter out low-confidence detections
    boxes = boxes[scores > 0.5]
    
    return boxes

def main(video_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model().to(device)

    cap = cv2.VideoCapture(video_path)
    _, first_frame = cap.read()

    trackers = []
    all_unique_ids = set()
    colors = generate_colors(100)  # Adjust the number based on the expected maximum number of objects

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        boxes = detect_people(frame, model, device)

        cost_matrix = np.zeros((len(trackers), len(boxes)))

        for i, tracker in enumerate(trackers):
            for j, box in enumerate(boxes):
                cost_matrix[i, j] = 1 - iou(tracker[-1], box)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        new_trackers = []

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 0.3:  # Threshold for matching
                trackers[i].append(boxes[j])
                color = colors[i % len(colors)]  # Assign a unique color based on object index
                cv2.rectangle(frame, (boxes[j][0], boxes[j][1]), (boxes[j][2], boxes[j][3]), color, 2)
            else:
                new_trackers.append(trackers[i])

        for k, box in enumerate(boxes):
            new_trackers.append([box])
            color = colors[len(trackers) + k % len(colors)]  # Assign a unique color based on object index
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

        trackers = new_trackers

        # Update the set of unique IDs across all frames
        unique_ids = set()
        for tracker in trackers:
            for box in tracker:
                unique_ids.add(id(box))

        all_unique_ids.update(unique_ids)

        cv2.imshow('Tracking', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

    total_people = len(all_unique_ids)
    print(f"Total People in the Video: {total_people}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'easy_9.mp4'
    main(video_path)