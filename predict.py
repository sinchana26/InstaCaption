
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

# Load Model, Tokenizer, and Image Features
model = load_model("captioning_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_length.pkl", "rb") as f:
    max_length = pickle.load(f)

# Load CNN for Feature Extraction
base_model = InceptionV3(weights='imagenet')
cnn_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)


# Image Feature Extraction Function
def process_image(image_path):
    img = Image.open(image_path).resize((299, 299))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img




# Caption Generation Function
def generate_caption(image_path):
    feature = extract_features(image_path)
    text = ["<start>"]

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], maxlen=max_length, padding='post')
        prediction = model.predict([feature, seq])

        word_id = np.argmax(prediction)
        word = tokenizer.index_word.get(word_id, "<UNK>")

        if word == "<end>":
            break  # Stop immediately on first <end>

        text.append(word)

    # Convert to a readable string, ensuring no residual <end>
    caption = " ".join(text[1:])  # Remove "<start>"
    caption = caption.replace(" end", "").strip()  # Remove leftover "end" words

    return caption


# Test the Model with a New Image
if __name__ == "__main__":
    test_image = "Crime_742.jpg"

    if not os.path.exists(test_image):
        print("❌ Image not found! Please enter a valid path.")
    else:
        caption = generate_caption(test_image)
        print("\n🖼️ Generated Caption:", caption)
