# Face Verification Project

This project is a face verification system that uses VGG Face model to compare two images and determine whether they are of the same person.

## Project Structure

- `align_face.py`: Contains the `align_face` function which aligns faces in images for better comparison.
- `crop_face.py`: Contains the `crop_face` function which crops faces from images.
- `verify.py`: Contains the `FaceVerification` class which is used to verify whether two images are of the same person.
- `tests.py`: Contains tests for the face verification system using the LFW (Labeled Faces in the Wild) dataset.
- `helper.py`: Contains helper functions used across the project.
- `weights/`: Contains the weights for the models used in the project.
- `data/`: Contains the LFW dataset and other data used in the project.

## Setup

1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Download the weights for the models from the links provided in the `weights/` directory and place them in the appropriate subdirectories.

## Usage

You can use the `FaceVerification` class in `verify.py` to verify faces. It is recommended to preprocess
the images before infering them. Here's a basic example:

```python
from verify import FaceVerification
from align_face import align_face
from crop_face import crop_face

crop_face(align_face("path_to_img1"), file_name="img1.jpg")
crop_face(align_face("path_to_img2"), file_name="img2.jpg")

verifier = FaceVerification()
is_same_person, cos_similarity, euclidean_distance = verifier.verify_face("path_to_img1", "path_to_img2")
print(is_same_person)
```

## Folder Structure
```
├── data/
│   ├── lfw/
│   │   ├── lfw_funneled/
│   │   │   ├── ...
│   │   ├── pairs.txt
│   │   ├── pairsDevTest.txt
│   │   └── pairsDevTrain.txt
│   └── preprocessed/
│   └── test_images/
└── weights/
│   ├── face_alignment/
│   ├── face_crop/
│   └── face_verification/
├── main.ipynb
├── other files...
```