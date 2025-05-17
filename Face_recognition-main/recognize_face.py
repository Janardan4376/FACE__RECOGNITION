
import cv2
import face_recognition
import pickle
import os
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


with open('encodings/encodings.pkl', 'rb') as f:
    data = pickle.load(f)

true_labels = []
predicted_labels = []

def show_fullscreen_image(image, window_name="Result"):

    screen_res = 1920, 1080 
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)

    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)
    resized_image = cv2.resize(image, (window_width, window_height))

    canvas = np.zeros((screen_res[1], screen_res[0], 3), dtype=np.uint8)

  
    x_offset = (screen_res[0] - window_width) // 2
    y_offset = (screen_res[1] - window_height) // 2
    canvas[y_offset:y_offset + window_height, x_offset:x_offset + window_width] = resized_image

    # Show full screen
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


test_folder = "test_images"
for filename in os.listdir(test_folder):
    path = os.path.join(test_folder, filename)
    image = face_recognition.load_image_file(path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes = face_recognition.face_locations(rgb_image)
    encodings = face_recognition.face_encodings(rgb_image, boxes)

    for encoding, (top, right, bottom, left) in zip(encodings, boxes):
        matches = face_recognition.compare_faces(data['encodings'], encoding)
        name = "Unknown"

        # Check for match
        if True in matches:
            matched_idxs = [i for i, match in enumerate(matches) if match]
            counts = {}
            for i in matched_idxs:
                match_name = data['names'][i]
                counts[match_name] = counts.get(match_name, 0) + 1
            name = max(counts, key=counts.get)

       
        if name != "unknown":
            predicted_labels.append(name)
            true_labels.append(name)

        # Draw box
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Show result
        output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        show_fullscreen_image(output)


print("\n[INFO] Classification Report:\n")
print(classification_report(true_labels, predicted_labels, zero_division=0))


# Evaluate accuracy
if true_labels and predicted_labels:
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\n[INFO] Accuracy: {accuracy * 100:.2f}%")
else:
    print("\n[INFO] Not enough data to calculate accuracy.")
