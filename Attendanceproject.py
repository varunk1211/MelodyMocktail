import pickle
import cv2
import face_recognition
import os
import numpy as np

pickle_file = "encodeFile.p"

# Load Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def run_face_recognition():
    count = 0  # Initialize count for marking attendance
    attendance_marked = set()  # To store names for which attendance is marked
    prev_face_locations = None
    movement_threshold = 20  # Minimum movement in pixels to consider as live
    motion_count = 0  # Track the number of frames the face is static
    blink_count = 0  # Track the number of blinks detected
    blink_threshold = 2  # Number of frames without detecting eyes to confirm a blink
    
    if not os.path.exists(pickle_file):
        print("No encodings found. Run `encode_faces.py` first to create encodings.")
        return

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("window", 1080, 780)

    # Load encodings
    with open(pickle_file, "rb") as file:
        encodeListKnownWithId = pickle.load(file)
        encodeListKnown, studentIds = encodeListKnownWithId

    print("Loaded student IDs:", studentIds)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame. Exiting.")
            break

        # Flip the image horizontally to correct the mirrored effect
        img = cv2.flip(img, 1)

        img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        faceLocations = face_recognition.face_locations(img_small)
        currentFrameEncodings = face_recognition.face_encodings(img_small, faceLocations)

        for encodeFace, faceLoc in zip(currentFrameEncodings, faceLocations):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            distance = face_recognition.face_distance(encodeListKnown, encodeFace)

            if len(distance) > 0:
                matchIndex = np.argmin(distance)

                if matches[matchIndex]:
                    name = studentIds[matchIndex]
                    color = (0, 255, 0)  # Green for known faces
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown faces

                # Draw bounding box around the face
                top, right, bottom, left = faceLoc
                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                cv2.rectangle(img, (left, top), (right, bottom), color, 3)

                # Check for movement (liveness detection)
                if prev_face_locations is not None:
                    # Compare previous and current face locations
                    face_movement = np.linalg.norm(np.array(faceLoc) - np.array(prev_face_locations))
                    
                    if face_movement < movement_threshold:
                        motion_count += 1  # Increment motion count for static face
                        cv2.putText(img, "Static image detected", (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        name = ""  # Don't display name if static image is detected
                    else:
                        motion_count = 0  # Reset motion count if face is moving
                        cv2.putText(img, "Live face detected", (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                prev_face_locations = faceLoc

                # Eye blinking detection
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_roi = gray[top:bottom, left:right]  # Region of interest for the face
                eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5)  # Detect eyes in the face region

                if len(eyes) == 0:  # Blink detected, as no eyes were found
                    blink_count += 1  # Increment blink count if eyes are not detected (indicating a blink)
                else:
                    blink_count = 0  # Reset blink count if eyes are detected

                if blink_count >= blink_threshold:
                    cv2.putText(img, "Blink detected", (left, top - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Draw bounding boxes around the detected eyes
                for (ex, ey, ew, eh) in eyes:
                    ex, ey, ew, eh = ex + left, ey + top, ew, eh  # Adjust the coordinates for the original face location
                    cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)  # Blue box for eyes
                print("Blink detected", blink_count, "counts motion", motion_count)

                # Display the name only if the image is live and blinking is detected
                if name and name != "Unknown" and motion_count < 20 and blink_count >= blink_threshold:  # Motion count threshold to reject static images
                    # Display the name below the bounding box
                    cv2.putText(img, name, (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Increment the count if this name is recognized
                    if name in studentIds:
                        count += 1
                    # # Mark attendance when count reaches 3 consecutive frames
                    if count == 2 and name not in attendance_marked:
                        print(f"Attendance marked for {name}")
                        attendance_marked.add(name)  # Mark attendance for the student
                        count = 0  # Reset count after marking attendance
                        cap.release()
                        cv2.destroyAllWindows()
                        return attendance_marked

        # Show the image
        cv2.imshow("window", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name=run_face_recognition()
    print(name)
