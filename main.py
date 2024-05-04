from align_face import align_face
from crop_face import crop_face
from verify import FaceVerification


if __name__ == "__main__":
    face_verification = FaceVerification()
    aligned_face = align_face("gau1.jpg")
    aligned_face = align_face("gau2.jpg")
    cropped_face = crop_face("gau1.jpg")
    cropped_face = crop_face("gau2.jpg")
    verified_face = face_verification.verify_face("data/cropped/fallen.jpg", "data/cropped/gau1.jpg")
    if verified_face:
        print("Faces match!")
    else:
        print("Faces do not match!")