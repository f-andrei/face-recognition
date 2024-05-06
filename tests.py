import pandas as pd
from align_face import align_face
from crop_face import crop_face
from verify import FaceVerification

# Tests using the LFW (labelled faces in the wild) dataset
# https://www.kaggle.com/datasets/atulanandjha/lfwpeople/data

def preprocess_lfw(pairs: pd.DataFrame, start: int = None, stop: int = None):
    for index, row in pairs[start:stop].iterrows():
        # Extract the image paths from the row
        pair = row.iloc[0].split('\t')

        # Check if the pair contains 3 or 4 elements
        if len(pair) == 3:
            person_name, img1_number, img2_number = pair
            img1_path = f"data/lfw/lfw_funneled/{person_name}/{person_name}_{img1_number.zfill(4)}.jpg"
            img2_path = f"data/lfw/lfw_funneled/{person_name}/{person_name}_{img2_number.zfill(4)}.jpg"
        else:
            person_name1, img1_number, person_name2, img2_number = pair
            img1_path = f"data/lfw/lfw_funneled/{person_name1}/{person_name1}_{img1_number.zfill(4)}.jpg"
            img2_path = f"data/lfw/lfw_funneled/{person_name2}/{person_name2}_{img2_number.zfill(4)}.jpg"
        

        # Align and crop face
        crop_face(align_face(img1_path))
        crop_face(align_face(img2_path))
    print("Finished preprocessing.")
        
def test_model_on_lfw(model: FaceVerification, pairs: pd.DataFrame, preprocessed: bool = False):
    data_list = []

    # Iterate over the first two rows of the pairs DataFrame
    for index, row in pairs.iterrows():
        results = {}
        
        # Extract the image paths from the row
        pair = row.iloc[0].split('\t')

        # Check if the pair contains 3 or 4 elements
        if len(pair) == 3:
            person_name, img1_number, img2_number = pair
            img1_path = f"data/lfw/lfw_funneled/{person_name}/{person_name}_{img1_number.zfill(4)}.jpg"
            img2_path = f"data/lfw/lfw_funneled/{person_name}/{person_name}_{img2_number.zfill(4)}.jpg"
        else:
            person_name1, img1_number, person_name2, img2_number = pair
            img1_path = f"data/lfw/lfw_funneled/{person_name1}/{person_name1}_{img1_number.zfill(4)}.jpg"
            img2_path = f"data/lfw/lfw_funneled/{person_name2}/{person_name2}_{img2_number.zfill(4)}.jpg"
        
        # Change the image paths if the images have been preprocessed
        if preprocessed:
            img1_path = f"data/preprocessed/{img1_path.split('/')[-1]}"
            img2_path = f"data/preprocessed/{img2_path.split('/')[-1]}"

        # Verify the faces
        is_same_person, cosine_similarity, euclidean_distance = model.verify_face(img1_path, img2_path)
        
        # Extract the person names from the image paths
        person_1 = img1_path.split("/")[-1]
        person_1 = person_1[:person_1.rfind("_")]
        person_2 = img2_path.split("/")[-1]
        person_2 = person_2[:person_2.rfind("_")]

        # Check if the person names are the same
        actual = person_1 == person_2
        
        # Add results to the dictionary
        results["Actual"] = actual
        results["Predicted"] = is_same_person
        results["Euclidean_distance"] = euclidean_distance
        results["Cos_similarity"] = cosine_similarity

        # Append the results to the data_list
        data_list.append(pd.DataFrame(results, index=[0]))

    # Concatenate data_list into a single DataFrame
    results_df = pd.concat(data_list)

    # Save the DataFrame to CSV
    results_df.to_csv("results.csv", index=False)  
    return results_df 

if __name__ == "__main__":
    pairs = pd.read_csv("data/lfw/pairsDevTest.txt")
    results = test_model_on_lfw(FaceVerification(), pairs, True)
    # print(preprocess(pairs))


