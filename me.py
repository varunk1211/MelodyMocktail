import face_recognition
import cv2
import numpy as np

# Load the known image
known_image = face_recognition.load_image_file("known_face.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Initialize known face data
known_faces = [known_encoding]
known_names = ["Person 1"]  # Change to the person's name

# Load an unknown image for recognition
unknown_image = face_recognition.load_image_file("unknown_face.jpg")
unknown_encodings = face_recognition.face_encodings(unknown_image)

# Process each detected face
for unknown_encoding in unknown_encodings:
    matches = face_recognition.compare_faces(known_faces, unknown_encoding)
    name = "Unknown"

    # Find the best match
    face_distances = face_recognition.face_distance(known_faces, unknown_encoding)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = known_names[best_match_index]

    print(f"Detected: {name}")

# If using a webcam, you can modify this script to process live video.
