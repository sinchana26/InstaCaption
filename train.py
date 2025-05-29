import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Add, Input, Dropout
from tensorflow.keras.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# === STEP 1: Load and Process Dataset ===
def load_captions(file_path):
    captions = {}
    df = pd.read_csv(file_path, quotechar='"', skipinitialspace=True)
    for _, row in df.iterrows():
        image_id, caption = row["image"], row["caption"].lower()
        if image_id not in captions:
            captions[image_id] = []
        captions[image_id].append("<start> " + caption + " <end>")
    return captions

dataset_path = "cleaned_captions.txt"  # Path to your captions file
image_dir = "Images"  # Directory containing images
captions = load_captions(dataset_path)

# Tokenize captions
all_captions = [cap for cap_list in captions.values() for cap in cap_list]
tokenizer = Tokenizer(num_words=5000, oov_token="<UNK>")
tokenizer.fit_on_texts(all_captions)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
max_length = max(len(seq.split()) for seq in all_captions)

# === STEP 2: Extract Image Features Using InceptionV3 ===
base_model = InceptionV3(weights='imagenet')
cnn_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def process_image(image_path):
    img = Image.open(image_path).resize((299, 299))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def extract_features(img_path):
    image = process_image(img_path)
    feature_vector = cnn_model.predict(image)
    return feature_vector.reshape((2048,))  # Fixes shape issue


# Extract and save features
image_features = {}
for img_name in tqdm(os.listdir(image_dir)):
    img_path = os.path.join(image_dir, img_name)
    try:
        image_features[img_name] = extract_features(img_path)
    except Exception as e:
        print(f"Skipping {img_name}: {e}")


with open("image_features.pkl", "wb") as f:
    pickle.dump(image_features, f)

# === STEP 3: Prepare Training Data ===
def create_training_data():
    X_image, X_text, y = [], [], []
    for key, caps in captions.items():
        if key in image_features:
            feature = image_features[key]
            for cap in caps:
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    X_image.append(feature)
                    X_text.append(seq[:i])
                    y.append(seq[i])
    return np.array(X_image), pad_sequences(X_text, maxlen=max_length), to_categorical(y, num_classes=vocab_size)

X_image, X_text, y = create_training_data()

# === STEP 4: Build Image Captioning Model ===
image_input = Input(shape=(2048,))
image_embedding = Dense(256, activation='relu')(image_input)

text_input = Input(shape=(max_length,))
text_embedding = Embedding(vocab_size, 256, mask_zero=True)(text_input)
text_lstm = LSTM(256)(text_embedding)

merged = Add()([image_embedding, text_lstm])
output = Dense(vocab_size, activation='softmax')(merged)

model = Model(inputs=[image_input, text_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# === STEP 5: Train the Model ===
model.fit([X_image, X_text], y, epochs=100, batch_size=32, verbose=1)

# === STEP 6: Generate Captions for New Images ===
def generate_caption(image_path):
    feature = extract_features(image_path).reshape((1, 2048))  # Fix feature shape
    text = "<start>"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], maxlen=max_length)  # Shape (1, max_length)
        prediction = model.predict([feature, seq])  # Ensure correct input shapes
        word = tokenizer.index_word.get(np.argmax(prediction), "<UNK>")
        if word == "<end>":
            break
        text += " " + word
    return text.replace("<start>", "").replace("<end>", "").strip()

# Test on an image
test_image = os.path.join(image_dir, "Crime_1.jpg")
caption = generate_caption(test_image)
print("Generated Caption:", caption)

# Save the trained model
model.save("captioning_model.keras")

# Save the tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save image features
with open("image_features.pkl", "wb") as f:
    pickle.dump(image_features, f)

# Save max_length
with open("max_length.pkl", "wb") as f:
    pickle.dump(max_length, f)

print("✅ Model, Tokenizer, and Image Features Saved Successfully!")
