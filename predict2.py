# import os
# import numpy as np
# import pickle
# import tensorflow as tf
# from tensorflow.keras.applications import InceptionV3
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from PIL import Image
# from gtts import gTTS
# from playsound import playsound
# import tempfile
# import threading
# from translate import Translator  # More reliable translation library
#
# # Load Model, Tokenizer, and Image Features
# model = load_model("captioning_model.keras")
#
# with open("tokenizer.pkl", "rb") as f:
#     tokenizer = pickle.load(f)
#
# with open("max_length.pkl", "rb") as f:
#     max_length = pickle.load(f)
#
# # Load CNN for Feature Extraction
# base_model = InceptionV3(weights='imagenet')
# cnn_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
#
#
# # Image Feature Extraction Function
# def process_image(image_path):
#     img = Image.open(image_path).resize((299, 299))
#     img = np.array(img) / 255.0
#     img = np.expand_dims(img, axis=0)
#     return img
#
#
# def extract_features(img_path):
#     image = process_image(img_path)
#     feature_vector = cnn_model.predict(image)
#     return feature_vector.reshape((1, 2048))
#
#
# # Caption Generation Function
# def generate_caption(image_path):
#     feature = extract_features(image_path)
#     text = ["<start>"]
#
#     for _ in range(max_length):
#         seq = tokenizer.texts_to_sequences([text])[0]
#         seq = pad_sequences([seq], maxlen=max_length, padding='post')
#         prediction = model.predict([feature, seq])
#
#         word_id = np.argmax(prediction)
#         word = tokenizer.index_word.get(word_id, "<UNK>")
#
#         if word == "<end>":
#             break  # Stop immediately on first <end>
#
#         text.append(word)
#
#     # Convert to a readable string, ensuring no residual <end>
#     caption = " ".join(text[1:])  # Remove "<start>"
#     caption = caption.replace(" end", "").strip()  # Remove leftover "end" words
#
#     return caption
#
#
# # Language Selection for TTS
# language_options = {
#     "1": ("English", "en"),
#     "2": ("Kannada", "kn"),
#     "3": ("Hindi", "hi"),
#     "4": ("Telugu", "te"),
#     "5": ("Tamil", "ta"),
# }
#
#
# # Function to Translate and Convert Text to Speech
# def text_to_speech(text, language):
#     translator = Translator(to_lang=language)
#     translated_text = translator.translate(text)
#
#     print(f"\n📝 Translated Caption in {language}: {translated_text}")
#
#     temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
#     tts = gTTS(text=translated_text, lang=language)
#     tts.save(temp_audio_file.name)
#
#     # Play the generated audio in a separate thread to avoid UI freezing
#     threading.Thread(target=playsound, args=(temp_audio_file.name,)).start()
#
#     return temp_audio_file.name
#
#
# # Main Execution
# if __name__ == "__main__":
#     print("\n📸 Image Caption Generator with Voice 🎤")
#     image_path = "Images/Crime_378.jpg"
#
#     if not os.path.exists(image_path):
#         print("❌ Image not found! Please enter a valid path.")
#         exit()
#
#     print("\n🌍 Select a language for speech output:")
#     for key, (lang_name, _) in language_options.items():
#         print(f"{key}. {lang_name}")
#
#     choice = input("\nEnter your choice (1-5): ")
#
#     if choice not in language_options:
#         print("❌ Invalid choice. Please select a valid option.")
#         exit()
#
#     selected_lang_name, selected_lang_code = language_options[choice]
#
#     print("\n📝 Generating caption...")
#     caption = generate_caption(image_path)
#     print(f"\n✅ Generated Caption: {caption}")
#
#     print("\n🔊 Translating and Playing in Selected Language...")
#     text_to_speech(caption, selected_lang_code)


import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
from gtts import gTTS
from playsound import playsound
import tempfile
import threading
from translate import Translator
import google.generativeai as genai  # Import Gemini API

# Configure Gemini API
genai.configure(api_key="AIzaSyDOgICUPFsiU5RBi9VfeAmYrTpDmNtxhps")  # Replace with your actual API key

# Load Model, Tokenizer, and Image Features
model = load_model("Final_captioning_model.keras")

with open("Final_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("Final_max_length.pkl", "rb") as f:
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


def extract_features(img_path):
    image = process_image(img_path)
    feature_vector = cnn_model.predict(image)
    return feature_vector.reshape((1, 2048))


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

    caption = " ".join(text[1:])  # Remove "<start>"
    caption = caption.replace(" end", "").strip()  # Remove leftover "end" words

    return caption


# Function to Rephrase Caption using Gemini API
def rephrase_caption(caption):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Rephrase the following sentence in exactly 30 words:\n\n'{caption}'"
    response = model.generate_content(prompt)
    return response.text.strip()


# Language Selection for TTS
language_options = {
    "1": ("English", "en"),
    "2": ("Kannada", "kn"),
    "3": ("Hindi", "hi"),
    "4": ("Telugu", "te"),
    "5": ("Tamil", "ta"),
}


# Function to Translate and Convert Text to Speech
def text_to_speech(text, language):
    translator = Translator(to_lang=language)
    translated_text = translator.translate(text)

    print(f"\n📝 Translated Caption in {language}: {translated_text}")

    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts = gTTS(text=translated_text, lang=language)
    tts.save(temp_audio_file.name)

    # Play the generated audio in a separate thread
    threading.Thread(target=playsound, args=(temp_audio_file.name,)).start()

    return temp_audio_file.name


# Main Execution
if __name__ == "__main__":
    print("\n📸 Image Caption Generator with Voice 🎤")
    image_path = "Images/Crime_378.jpg"

    if not os.path.exists(image_path):
        print("❌ Image not found! Please enter a valid path.")
        exit()

    print("\n🌍 Select a language for speech output:")
    for key, (lang_name, _) in language_options.items():
        print(f"{key}. {lang_name}")

    choice = input("\nEnter your choice (1-5): ")

    if choice not in language_options:
        print("❌ Invalid choice. Please select a valid option.")
        exit()

    selected_lang_name, selected_lang_code = language_options[choice]

    print("\n📝 Generating caption...")
    caption = generate_caption(image_path)
    print(f"\n✅ Generated Caption: {caption}")

    print("\n🔄 Rephrasing caption using Gemini API...")
    rephrased_caption = rephrase_caption(caption)
    print(f"\n✏️ Rephrased Caption: {rephrased_caption}")

    print("\n🔊 Translating and Playing in Selected Language...")
    text_to_speech(rephrased_caption, selected_lang_code)
