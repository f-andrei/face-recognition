import cv2
import numpy as np
from PIL import Image
import dlib
import os
from helper import *


def load_model():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("weights/face_alignment/shape_predictor_5_face_landmarks.dat")
    return detector, predictor

folder_path = "data/test_images/"
destination_folder = "data/aligned"


def align_face(image_file):
    detector, predictor = load_model()
    image_path = os.path.join(folder_path, image_file)
    try:
        img = cv2.imread(image_path)

        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        detected_faces = detector(grayscale_image, 0)
        
        # If no face is detected, delete the image and return
        if len(detected_faces) == 0:
            # os.remove(f"{folder_path}/{img_path}")
            return f"no face detected for {image_path}"

        # Get the face landmarks
        face_landmarks = predictor(grayscale_image, detected_faces[0])

        # Normalize the face landmarks
        face_landmarks = shape_to_landmark_list(face_landmarks)

        # Get the eyes and nose
        nose, l_eye, r_eye = get_eyes_nose(face_landmarks)

        # Calculate the rotation angle
        forehead_center = ((l_eye[0] + r_eye[0]) // 2, (l_eye[1] + r_eye[1]) // 2)

        # Get the center of the face
        face_center = (int((detected_faces[0].left() + detected_faces[0].right()) / 2), 
                        int((detected_faces[0].top() + detected_faces[0].top()) / 2))
        
        # Calculate the lengths of the lines
        line1 = eucliean_distance(forehead_center, nose)
        line2 = eucliean_distance(face_center, nose)
        line3 = eucliean_distance(face_center, forehead_center)
        
        # Calculate the cosine of the angle
        cos_a = cosine_formula(line1, line2, line3)

        # Calculate the rotation angle
        rotation_angle = np.arccos(cos_a)

        # Rotate the nose tip
        rotated_nose_tip = rotate_point(nose, forehead_center, rotation_angle)
        rotated_nose_tip = (int(rotated_nose_tip[0]), int(rotated_nose_tip[1]))

        # Check if the nose tip is between the forehead center, face center, and the nose tip
        if is_between(nose, forehead_center, face_center, rotated_nose_tip):
            rotation_angle = np.degrees(-rotation_angle)
        else:
            rotation_angle = np.degrees(rotation_angle)

        # Rotate the image
        img = Image.fromarray(img)
        img = np.array(img.rotate(rotation_angle))

        # Save the aligned image)
        destination_path = os.path.join(destination_folder, image_file)
        cv2.imwrite(destination_path, img)
        return destination_path
    except Exception as e:
        print(f"Error: {e}")
    print("finished aligning faces")
