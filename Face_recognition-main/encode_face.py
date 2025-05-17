# encode_faces.py

import face_recognition
import os
import pickle

dataset_path = 'dataset'
encodings_path = 'encodings/encodings.pkl'

known_encodings = []
known_names = []

# Loop through each person folder
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    # Loop through each image of the person
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        image = face_recognition.load_image_file(image_path)

        # Detect and encode
        face_locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person)

# Save encodings
with open(encodings_path, 'wb') as f:
    pickle.dump({'encodings': known_encodings, 'names': known_names}, f)

print(f"[INFO] Encoded {len(known_encodings)} faces.")
