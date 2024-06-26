from keras.models import Model, Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.imagenet_utils import preprocess_input
from typing import Tuple

# Define the VGG Face model
model = Sequential()

# Add the layers to the model
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))


class FaceVerification():
    def __init__(self) -> None:
        self._model = model
        self._model.load_weights("weights/face_verification/vgg_face_weights.h5")
        self._face_descriptor = Model(inputs=self._model.layers[0].input, outputs=self._model.layers[-2].output)

    def _preprocess_image(self, image: np.ndarray):
        image = load_img(image, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        return image

    def _find_cos_similarity(self, source_representation, test_representation) -> float:
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def _find_euclidean_distance(self, source_representation, test_representation) -> float:
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    def verify_face(self, img1_path: str, img2_path: str, epsilon: float = 0.3147) -> Tuple[bool, float, float]:
        # Get the representations of the images
        img1_representation = self._face_descriptor.predict(self._preprocess_image(img1_path))[0,:]
        img2_representation = self._face_descriptor.predict(self._preprocess_image(img2_path))[0,:]
        
        # Find the cosine similarity and euclidean distance
        cosine_similarity = self._find_cos_similarity(img1_representation, img2_representation)
        euclidean_distance = self._find_euclidean_distance(img1_representation, img2_representation)
        
        # Check if the images are of the same person
        if cosine_similarity < epsilon: return True, cosine_similarity, euclidean_distance
        return False, cosine_similarity, euclidean_distance
    
