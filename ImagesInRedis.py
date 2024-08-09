import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
import msgpack_numpy as m
import numpy as np
import redis
import os
import uuid
from PIL import Image

# Load the pre-trained ResNet50 model + higher level layers
base_model = ResNet50(weights='imagenet', include_top=False)

# Add custom layers to reduce the output to 1024 dimensions
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)

# This is the model we will use for feature extraction
model = Model(inputs=base_model.input, outputs=x)

# Debug: Check the model output shape
print(f"Model output shape: {model.output_shape}")

def get_image_feature_vector(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    
    feature_vector = model.predict(img_data)
    feature_vector = np.squeeze(feature_vector)
    return feature_vector

def store_numpy_array(redis_client, array, img_path):
    key = str(uuid.uuid4())  # Generate a unique UUID
    data = {
        'vector': m.packb(array),
        'img_path': img_path
    }
    redis_client.set(key, m.packb(data))
    return key

def retrieve_numpy_array(redis_client, key):
    data = m.unpackb(redis_client.get(key))
    return m.unpackb(data['vector']), data['img_path']

def process_and_store_images(folder_path, redis_client):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            feature_vector = get_image_feature_vector(img_path)
            print(feature_vector.shape)
            key = store_numpy_array(redis_client, feature_vector, img_path)
            print(f"Stored feature vector for {filename} with UUID {key}")

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def find_most_similar_image(redis_client, query_vector):
    max_similarity = -1
    most_similar_image = None
    most_similar_img_path = None
    
    for key in redis_client.keys():
        stored_vector, img_path = retrieve_numpy_array(redis_client, key)
        similarity = cosine_similarity(query_vector, stored_vector)
        
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_image = key
            most_similar_img_path = img_path

    return most_similar_image, max_similarity, most_similar_img_path

# Initialize Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Paths to the folders containing cat and dog images
cat_folder_path = "C:/Users/HTR036/OneDrive - Maersk Group/Pictures/Screenshots/cat"
dog_folder_path = "C:/Users/HTR036/OneDrive - Maersk Group/Pictures/Screenshots/dog"

# Process and store images from both folders
process_and_store_images(cat_folder_path, redis_client)
process_and_store_images(dog_folder_path, redis_client)

print(redis_client.keys())
# Example usage to find the most similar image
new_image_path = "C:/Users/HTR036/OneDrive - Maersk Group/Pictures/Screenshots/newcat.jpg"
query_vector = get_image_feature_vector(new_image_path)
most_similar_image, similarity_score, most_similar_img_path = find_most_similar_image(redis_client, query_vector)

if similarity_score * 100 > 65:
    print(f"The most similar image is: {most_similar_image}")
    print(f"Similarity score: {similarity_score}")
    
    # Display the most similar image
    similar_image = Image.open(most_similar_img_path)
    similar_image.show()
else:
    print("No image found with sufficient similarity")
    new_key = store_numpy_array(redis_client, query_vector, new_image_path)
    print(f"Stored new image with UUID {new_key}")
