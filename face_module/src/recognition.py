import os
import pickle
import face_recognition


# Project folder
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model file
encoding_file = os.path.join(base_path, "models", "encodings.pkl")


# Load known face encodings
with open(encoding_file, "rb") as file:
    data = pickle.load(file)


known_encodings = data["encodings"]
known_names = data["names"]


print("Known faces loaded:", len(known_encodings))


# Test image
image_path = os.path.join(base_path, "test.jpg")


# Check image
if not os.path.exists(image_path):
    print("Test image not found!")
    exit()


# Load classroom photo
image = face_recognition.load_image_file(image_path)


# Find all faces in the image
face_locations = face_recognition.face_locations(image)

print("Faces detected:", len(face_locations))


# Generate encodings for all detected faces
face_encodings = face_recognition.face_encodings(
    image,
    face_locations
)


recognized_names = []


# Recognize every detected face
for face_encoding in face_encodings:

    # Calculate distance from every known face
    face_distances = face_recognition.face_distance(
        known_encodings,
        face_encoding
    )

    # Find the closest known face
    best_match_index = face_distances.argmin()

    # Default name
    name = "Unknown"

    # Check if the closest face is actually close enough
    if face_distances[best_match_index] < 0.50:
        name = known_names[best_match_index]

    print("Best distance:", face_distances[best_match_index])
    print("Recognized as:", name)

    recognized_names.append(name)


# Display results
print("\nRecognition Results:")

for name in recognized_names:
    print(name)