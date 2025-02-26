def run_face_recognition():
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
                    text = f"{name}"
                    color = (0, 255, 0)  # Green for known faces
                else:
                    text = "Unknown"
                    color = (0, 0, 255)  # Red for unknown faces

                # Draw bounding box
                top, right, bottom, left = faceLoc
                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                cv2.rectangle(img, (left, top), (right, bottom), color, 3)
                cv2.putText(img, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("window", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run encoding first, then recognition
if not os.path.exists(pickle_file):
    print("No encodings found. Creating now...")
    encode_faces()
else:
    print("Encodings already exist. Skipping encoding.")

# Start face recognition
run_face_recognition()