import os
import cv2
import numpy as np

folder_path = "data/aligned/"
destination_folder = "data/cropped"

def load_model():
    CAFFE_MODEL = "weights/face_crop/res10_300x300_ssd_iter_140000.caffemodel"
    PROTO_TXT = "weights/face_crop/deploy.prototxt"
    detector = cv2.dnn.readNetFromCaffe(PROTO_TXT, CAFFE_MODEL)
    return detector

def crop_face(image_file):
    try:
        detector = load_model()
        image_path = os.path.join(folder_path, image_file)
        # Load image
        img = cv2.imread(image_path)
        (h, w) = img.shape[:2]

        # Create image blob (prepro)
        # image_blob = cv2.dnn.blobFromImage(image=img)
        image_blob = cv2.dnn.blobFromImage(cv2.resize(img, (224, 224)), 1.0, (224, 224))

        # Load image blob into model
        detector.setInput(image_blob)

        # Make predictions
        detections = detector.forward()

        # Access the first detection (assuming there's only one face in the image)
        detection = detections[0, 0][0]

        # Extract confidence score from the detection
        confidence = detection[2]

        # Check if the confidence score is above the threshold
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = img[startY:endY, startX:endX]
                # face = cv2.resize(face, (224, 224))
                destination_path = os.path.join(destination_folder, image_file)
                cv2.imwrite(destination_path, face)
                return destination_path

    except Exception as e:
        print(e)
