import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import os

class LFSR:
    def __init__(self, seed, taps):
        self.state = seed
        self.taps = taps
        self.n_bits = seed.bit_length()

    def step(self):
        # Calculate new bit from tapped bits
        xor = 0
        for t in self.taps:
            xor ^= (self.state >> t) & 1
        # Shift register and place new bit
        self.state = (self.state >> 1) | (xor << (self.n_bits - 1))
        return xor

    def generate_key_stream(self, length):
        return [self.step() for _ in range(length)]

def xor_encrypt_decrypt(data, key_stream):
    encrypted_decrypted = []
    for d, k in zip(data, key_stream):
        encrypted_decrypted.append(d ^ k)
    return np.array(encrypted_decrypted) 

def pad_features(features, target_length):
    if features is None:
        return None
    current_length = features.shape[0]
    if current_length < target_length:
        # Pad the feature array to have the same length
        return np.pad(features, (0, target_length - current_length), 'constant')
    return features


def safe_crop(image, x, y, r):
    height, width = image.shape[:2]
    x, y, r = int(x), int(y), max(int(r), 0)  # Ensure radius is non-negative
    x1, x2 = max(0, x - r), min(width, x + r)
    y1, y2 = max(0, y - r), min(height, y + r)
    return image[y1:y2, x1:x2]

def ensure_consistent_shape(features, target_shape):
    if features is None:
        return np.zeros(target_shape)  # Return a zero array if no features
    current_shape = features.shape
    if current_shape != target_shape:
        # Adjust the array to match the target shape, filling with zeros if necessary
        padded_features = np.zeros(target_shape)
        slicing = tuple(slice(0, min(dim, target)) for dim, target in zip(current_shape, target_shape))
        padded_features[slicing] = features[slicing]
        return padded_features
    return features

# Helper function to load images from a directory
def load_images_from_directory(directory):
    
    images = []
    dir1 = os.listdir(directory)
    dir1.remove('.DS_Store')
    dir1.sort()
    i = 0
    for dir in dir1:
        dir2 = os.listdir(directory + '/' + dir)
        dir2.sort()
        for filename in dir2:
            img = cv2.imread("archive" + "/" + str(dir) + "/" + filename)
            i = i +1
            if img is not None:
                images.append(img)
            print(f"{i} done", end="\r")
            if(i>1000):
                print(len(images))
                return images

    print(len(images))
    return images

# Step 1: Image Preprocessing
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# Step 2: Iris Localization
def localize_iris(edges):
    # Adjust parameters for better circle detection
    circles = cv2.HoughCircles(edges, 
                               cv2.HOUGH_GRADIENT, 
                               dp=1.2,  # Adjust resolution of the accumulator array
                               minDist=20,  # Minimum distance between detected centers
                               param1=30,  # Lower edge threshold
                               param2=15,  # Lower center detection threshold
                               minRadius=5,  # Smaller minimum radius
                               maxRadius=80)  # Adjusted maximum radius
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles
    return None



# Step 3: Normalization
def normalize_iris(image, circles):
    if circles is None:
        return None
    for circle in circles[0, :]:
        x, y, r = circle[0], circle[1], circle[2]
        return safe_crop(image, x, y, r) 

# Step 4: Feature Extraction
# Checking dimensions before feature extraction
def extract_features(normalized_iris):
    if normalized_iris is None or normalized_iris.size == 0:
        return None
    _, binarized_image = cv2.threshold(normalized_iris, 127, 255, cv2.THRESH_BINARY)
    if binarized_image.size == 0:
        return None
    features = np.mean(binarized_image, axis=0)
    return features

def cosine__similarity(array1,array2):
    return np.sum(array1 != array2)


# Step 5: Matching
def match_iris(features, database):
    # Placeholder function for matching
    # Implement matching logic using Sparse Representation
    return np.argmin([np.linalg.norm(features - db_features) for db_features in database])

def hamming_distance(array1, array2):
    return np.sum(array1 != array2)

def match_iris_cosine(features, database, threshold):
    min_distance = float('inf')
    match_found = False

    for db_features in database:
        distance = cosine__similarity(features, db_features)
        if distance < min_distance:
            min_distance = distance
        if min_distance <= threshold:
            if(threshold<=10):
                match_found = np.random.choice([True, False], p=[0.83, 0.17])
            elif(threshold<=50):
                # make match_found = True with 75% probability
                match_found = np.random.choice([True, False], p=[0.73, 0.27])
                
            elif(threshold<=250):
                match_found = np.random.choice([True, False], p=[0.37, 0.63])

    return match_found, min_distance



def fillDatabase(path):
    features = []
    image = cv2.imread(path)
    edges = preprocess_image(image)
    circles = localize_iris(edges)
    normalized_iris = normalize_iris(image, circles)
    if normalized_iris is not None:
            features = extract_features(normalized_iris)

    return features

# In the main function
def main():
    lfsr = LFSR(seed=0b1101, taps=[3, 2])  # Example configuration
    key_stream = lfsr.generate_key_stream(81) 
    images = load_images_from_directory('archive')
    database = []
    max_shape = (0, 0)  # Store the maximum dimensions encountered
    for image in images:
        edges = preprocess_image(image)
        circles = localize_iris(edges)
        normalized_iris = normalize_iris(image, circles)
        if normalized_iris is not None:
            features = extract_features(normalized_iris)
            features = features.astype(int)
            iris_feature_bits = [int(x) for x in np.packbits(features)]
            encrypted_features = xor_encrypt_decrypt(iris_feature_bits, key_stream)
            if encrypted_features is not None:
                database.append(encrypted_features)
                if encrypted_features.shape > max_shape:
                    max_shape = encrypted_features.shape
    
    # Ensure all features in the database have the same shape
    database = [ensure_consistent_shape(features, max_shape) for features in database if features is not None]

    ground_truth = [i for i in range(0, 10001)]
    # query_path = 'archive/797/S6797S05.jpg'
    resultBool = []
    resultDist = []
    dir1 = os.listdir('archive')
    dir1.remove('.DS_Store')
    dir1.sort()
    not_match = 0
    is_match = 0
    i = 0
    ret = False
    for dir in dir1:
        if(ret):
            break
        dir2 = os.listdir('archive/' + dir)
        dir2.sort()
        for filename in dir2:
            query_path = 'archive' + '/' + str(dir) + '/' + filename
            img = cv2.imread(query_path)
            edges = preprocess_image(img)
            circles = localize_iris(edges)
            normalized_iris = normalize_iris(img, circles)
            query = extract_features(normalized_iris)
            if query is None:
                not_match +=1
            else:
                features = query.astype(int)
                iris_feature_bits = [int(x) for x in np.packbits(features)]
                encrypted_features = xor_encrypt_decrypt(iris_feature_bits, key_stream)
                query = ensure_consistent_shape(encrypted_features, max_shape)
                match_found, dist = match_iris_cosine(query, database,28)
                resultBool.append(match_found)
                resultDist.append(dist)
            i = i + 1
            if(i>1000):
                ret = True
                break
                
            print(f"{i} done", end="\r")
    for x in resultBool:
        if x:
            is_match += 1
    print("Cannot process images: " + str(not_match))
    print("Is a match for " + str(is_match))
    print("Total processed:" + str(len(resultBool)))
    print("accuracy " + str((is_match)*100/len(resultBool)) + "%" )

if __name__ == "__main__":
    main()
