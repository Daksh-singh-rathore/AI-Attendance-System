print("FILE STARTED")

import os
import pickle
import face_recognition

# Project folder
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset and models folder
dataset_path = os.path.join(base_path, "dataset")
models_path = os.path.join(base_path, "models")


print("Face encoding started...")
print("Dataset path:", dataset_path)


known_encodings = []
known_names = []


# Check dataset folder
if not os.path.exists(dataset_path):
    print("Dataset folder not found!")
    exit()


# Read each person's folder
for person_name in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"\nProcessing: {person_name}")


    # Read images of the person
    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        print(f"Reading: {image_name}")


        # Load image
        image = face_recognition.load_image_file(image_path)


        # Generate face encoding
        encodings = face_recognition.face_encodings(image)


        # Check if face was found
        if encodings:

            known_encodings.append(encodings[0])
            known_names.append(person_name)

            print(f"Encoded successfully: {person_name}")

        else:

            print("No face found!")


# Create models folder if it doesn't exist
os.makedirs(models_path, exist_ok=True)


# Data to save
data = {
    "encodings": known_encodings,
    "names": known_names
}


# Save encodings
encoding_file = os.path.join(models_path, "encodings.pkl")

with open(encoding_file, "wb") as file:
    pickle.dump(data, file)


print("\nFace encoding completed!")
print("Total encodings:", len(known_encodings))
print("Saved at:", encoding_file)
