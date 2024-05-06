import os
import cv2
import numpy as np


DESTINATION_FOLDER = "data/preprocessed"

if not os.path.exists(DESTINATION_FOLDER):
	os.makedirs(DESTINATION_FOLDER)


def load_model() -> cv2.dnn_Net:
    CAFFE_MODEL = "weights/face_crop/res10_300x300_ssd_iter_140000.caffemodel"
    PROTO_TXT = "weights/face_crop/deploy.prototxt"
    detector = cv2.dnn.readNetFromCaffe(PROTO_TXT, CAFFE_MODEL)
    return detector


def crop_face(image: np.ndarray, file_name: str) -> str:
	try:
		detector = load_model()

		# Load image
		# img = cv2.imread(image_path)
		(h, w) = image.shape[:2]

		# Create image blob (prepro)
		image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0, (224, 224))

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

				# Crop the face
				face = image[startY:endY, startX:endX]

				target_height = 224
				target_width = 224

				# Calculate padding amounts
				face_height, face_width = face.shape[:2]
				scale_factor = min(target_height / face_height, target_width / face_width)
				new_face_height = int(face_height * scale_factor)
				new_face_width = int(face_width * scale_factor)
				top_pad = (target_height - new_face_height) // 2
				bottom_pad = target_height - new_face_height - top_pad
				left_pad = (target_width - new_face_width) // 2
				right_pad = target_width - new_face_width - left_pad

				# Resize and pad the face
				resized_face = cv2.resize(face, (new_face_width, new_face_height))
				padded_face = cv2.copyMakeBorder(resized_face, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])

				# Save the cropped face
				destination_path = os.path.join(DESTINATION_FOLDER, file_name)
				cv2.imwrite(destination_path, padded_face)
				
		return destination_path
	except AttributeError as e:
		print(f"Error: {e}")
		return file_name
	except ZeroDivisionError as e:
		print(f"Error: {e}")
		return file_name
	