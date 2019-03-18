# Number Of faces
# Fae Locations

from PIL import Image
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("donald_trump.jpg")

# Find all the faces in the image using the default HOG-based model.
# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
# See also: find_faces_in_picture_cnn.py
face_locations = face_recognition.face_locations(image)

print("I found {} face(s) in this photograph.".format(len(face_locations)))

for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()


# Checking if A Person Exists Within an Image

import face_recognition

# Load in our reference image of Joe Biden
known_image = face_recognition.load_image_file("donald_trump.jpg")
# Load in our image of a group of people
unknown_image = face_recognition.load_image_file("donald_trump1.jpg")

# Create a biden encoding
biden_encoding = face_recognition.face_encodings(known_image)[0]
# create an encoding based off our group photo
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare the encodings and try to determine if Biden exists within a photo
results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
# Print the results
print(results)