import sqlite3
import google.generativeai as genai  # Import Gemini API
import pickle
import tensorflow as tf
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.sequence import pad_sequences
from PIL import Image
from translate import Translator
from gtts import gTTS
import tempfile
import threading
from playsound import playsound
import streamlit as st
from Image import ImageCaptionGenerator


st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background-color: {"#2c77a1"};
            color: {"#FFFFFF"};
        }}
        </style>
        """,
        unsafe_allow_html=True
)



# Sidebar Content for Legal and Forensic Application
st.sidebar.title("AI-Driven Image Captioning")
st.sidebar.image("s1.png")
st.sidebar.markdown("Streamline legal and forensic evidence documentation with AI!")

st.sidebar.subheader("What is this Tool?")
st.sidebar.markdown(
    """
    This tool uses advanced AI to generate detailed captions for images, 
    assisting professionals in the legal and forensic fields with evidence documentation.
    """
)

st.sidebar.subheader("Key Features")
st.sidebar.markdown(
    """
    - **Effortless Documentation**: Automatically caption images like crime scenes, evidence, and legal documents.
    - **Time-Saving**: Minimize manual annotation efforts and improve consistency.
    - **Accurate Descriptions**: Generate contextually relevant captions for effective evidence presentation.
    - **Customizable Settings**: Tailor the captioning process to fit case-specific requirements.
    """
)

st.sidebar.subheader("Who Can Benefit?")
st.sidebar.markdown(
    """
    - **Legal Professionals**: Document and present evidence with precision.
    - **Forensic Experts**: Organize crime scene images and annotations seamlessly.
    - **Law Enforcement**: Streamline the preparation of visual reports for investigations.
    """
)

st.sidebar.subheader("Why Choose This Tool?")
st.sidebar.markdown(
    """
    - Enhance accuracy and efficiency in legal documentation.
    - Make visual data more accessible and understandable.
    - Utilize the power of AI to support critical decision-making in legal and forensic scenarios.
    """
)

st.sidebar.info("Upload images and let AI assist you in generating professional, accurate captions.")

st.image("coverpage.png")


# SQLite database setup
db_path = "users.db"

def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            police_id TEXT UNIQUE,
            password TEXT,
            location TEXT,
            station TEXT
        )
    """)
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Streamlit session state initialization
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "image_path" not in st.session_state:
    st.session_state["image_path"] = None
if "caption" not in st.session_state:
    st.session_state["caption"] = None
if "translated_text" not in st.session_state:
    st.session_state["translated_text"] = None


# User Authentication Functions
def register_user(username, police_id, password, location, station):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO users (username, police_id, password, location, station)
            VALUES (?, ?, ?, ?, ?)
        """, (username, police_id, password, location, station))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM users WHERE username = ? AND password = ?
    """, (username, password))
    user = cursor.fetchone()
    conn.close()
    return user



if not st.session_state["logged_in"]:
    st.sidebar.subheader("Login or Register")

    choice = st.sidebar.radio("Choose an option", ["Login", "Register"])
    if choice == "Register":
        username = st.sidebar.text_input("Username")
        police_id = st.sidebar.text_input("Police ID")
        password = st.sidebar.text_input("Password", type="password")
        location = st.sidebar.text_input("Location")
        station = st.sidebar.text_input("Station")
        if st.sidebar.button("Register"):
            if register_user(username, police_id, password, location, station):
                st.sidebar.success("Registration successful! Please log in.")
            else:
                st.sidebar.error("Registration failed. Username or Police ID may already exist.")
    elif choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            user = authenticate_user(username, password)
            if user:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.sidebar.success(f"Welcome, {username}!")
            else:
                st.sidebar.error("Invalid credentials.")

    st.markdown("""
        # **InceptionV3-Based Image Captioning with Text-to-Speech Application**

        ## **Introduction**
        In the field of law enforcement, analyzing and documenting images efficiently is crucial. 
        This AI-powered application helps **police personnel generate captions for images** and **convert captions into speech** in multiple languages.  
        The system leverages **InceptionV3** for image feature extraction and an **LSTM-based model** for caption generation.  
        Additionally, a **text-to-speech (TTS) feature** allows officers to listen to captions in different languages.

        ---

        ## **Features**

        ### 1. **User Authentication**
        - **Secure Registration & Login**:
          - Users (police personnel) must register with details such as **username, Police ID, password, location, and station**.
          - Secure authentication ensures only authorized users can access the system.
        - **Session Management**:
          - Users stay logged in during active sessions, ensuring **seamless interaction** with the application.

        ---

        ### 2. **Image Captioning with InceptionV3**
        - **AI-Powered Captioning**:
          - Uses **InceptionV3 for feature extraction** from images.
          - Features are passed into an **LSTM-based deep learning model** to generate textual captions.
        - **Deep Learning Approach**:
          - Features are extracted using **InceptionV3 pre-trained on ImageNet**.
          - A **custom LSTM-based captioning model** is trained on image-text pairs.
        - **Accurate Descriptions**:
          - Captions provide **meaningful descriptions** of uploaded images.
          - Particularly useful for **crime scene analysis** and **field reporting**.

        ---

        ### 3. **Text-to-Speech Conversion**
        - **Multilingual Audio Support**:
          - Converts generated captions into speech in **multiple languages**:
            - **English**
            - **Kannada**
            - **Hindi**
            - **Telugu**
            - **Tamil**
        - **Automated Translation**:
          - Uses **Google Translator API** to translate captions before speech conversion.
        - **Audio Playback**:
          - Users can **listen to captions directly** within the application.

        ---

        ## **Technologies Used**

        ### **Frontend Framework**
        - **Streamlit**:
          - Provides a user-friendly interface for uploading images and selecting language options.

        ### **Backend: Image Processing & Captioning**
        - **InceptionV3**:
          - Extracts **2048-dimensional feature vectors** from images.
        - **LSTM-based Captioning Model**:
          - Generates **context-aware captions** from extracted features.
          - Trained on a **large-scale dataset of images and captions**.
        - **Google Translator API**:
          - Enables **automatic translation** of captions before text-to-speech conversion.
        - **gTTS (Google Text-to-Speech)**:
          - Converts translated text into **natural-sounding speech**.

        ### **Database**
        - **SQLite**:
          - Stores user credentials securely for **authentication and access control**.

        ---

        ## **How It Works**

        ### **Step 1: User Authentication**
        1. New users **register** by providing necessary details.
        2. Registered users **log in securely**.
        3. The system verifies credentials via the **SQLite database**.

        ### **Step 2: Image Upload**
        1. Users **upload images** in JPG, JPEG, or PNG format.
        2. The system processes the image using **InceptionV3**.

        ### **Step 3: Caption Generation**
        1. **InceptionV3 extracts image features** (2048-dimensional vectors).
        2. The **LSTM model generates a descriptive caption**.

        ### **Step 4: Language Selection**
        1. Users select a **preferred language**.
        2. Captions are translated using the **Google Translator API**.

        ### **Step 5: Text-to-Speech Conversion**
        1. Translated captions are **converted into audio** using `gTTS`.
        2. The audio file is **played directly** within the application.

        ---

        ## **Benefits and Use Cases**

        ### **1. Enhanced Communication**
        - Provides **accurate image descriptions** for law enforcement teams.
        - **Overcomes language barriers** with multilingual support.

        ### **2. Time Efficiency**
        - Automates **image analysis and caption generation**, reducing workload.

        ### **3. Accessibility**
        - Converts **text-based captions into speech**, making it useful for visually impaired personnel.

        ### **4. Field Applications**
        - **Crime scene analysis**: Generate captions for forensic evidence.
        - **Surveillance image documentation**: Helps in **security monitoring**.
        - **Missing person reports**: Assists in **case documentation**.

        ---

        ## **Security and Scalability**
        - User credentials are **securely stored in SQLite**, ensuring data privacy.
        - Optimized **InceptionV3 feature extraction** for **faster performance**.
        - **Scalable model** allows **real-time caption generation**.

        ---
        """)
else:
    from Image import ImageCaptionGenerator

    # Configure Gemini API
    genai.configure(api_key="AIzaSyDxF9DN1o-QQkFlbF_Hc_7Yyy9RJRtDwbE")

    gemini_model = genai.GenerativeModel("gemini-1.5-pro")

    # Load Model, Tokenizer, and Image Features
    model = tf.keras.models.load_model("Final_captioning_model.keras")

    with open("Final_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("Final_max_length.pkl", "rb") as f:
        max_length = pickle.load(f)

    # Load CNN for Feature Extraction
    base_model = InceptionV3(weights='imagenet')
    cnn_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)


    # Image Feature Extraction Function
    def process_image(image):
        image = image.resize((299, 299))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image


    def extract_features(image):
        image = process_image(image)
        feature_vector = cnn_model.predict(image)
        return feature_vector.reshape((1, 2048))


    # Caption Generation Function
    def generate_caption(image):
        feature = extract_features(image)
        text = ["<start>"]

        for _ in range(max_length):
            seq = tokenizer.texts_to_sequences([text])[0]
            seq = pad_sequences([seq], maxlen=max_length, padding='post')
            prediction = model.predict([feature, seq])
            word_id = np.argmax(prediction)
            word = tokenizer.index_word.get(word_id, "<UNK>")

            if word == "<end>":
                break

            text.append(word)

        caption = " ".join(text[1:])
        caption = caption.replace(" end", "").strip()

        return caption


    # Rephrase Caption using Gemini API
    def rephrase_caption(caption):
        rephrase_model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Rephrase the following caption into exactly 30 words without using any punctuation except . and keep it deterministic.\n\n'{caption}'"
        response = rephrase_model.generate_content(prompt)
        rephrased_caption = response.text.strip().replace(",", "").replace(";", "").replace(":", "").replace("-",
                                                                                                             "").replace(
            "!", "").replace("?", "")
        if not rephrased_caption.endswith("."):
            rephrased_caption += "."
        return rephrased_caption


    # Language Selection for TTS
    language_options = {
        "English": "en",
        "Kannada": "kn",
        "Hindi": "hi",
        "Telugu": "te",
        "Tamil": "ta",
    }


    def text_to_speech(text, language):
        translator = Translator(to_lang=language)
        translated_text = translator.translate(text)
        st.write(f"**Translated Caption in {language}:** {translated_text}")
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts = gTTS(text=translated_text, lang=language)
        tts.save(temp_audio_file.name)
        threading.Thread(target=playsound, args=(temp_audio_file.name,)).start()
        return temp_audio_file.name


    # Additional Caption Generation Function
    def generate_caption_2(image):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_file.name)
        temp_file.close()
        caption_generator = ImageCaptionGenerator()
        caption = caption_generator.generate_caption(temp_file.name)
        return rephrase_caption(caption)


    # Streamlit UI
    st.write("Upload an image and get an AI-generated caption, with translation and speech synthesis!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    selected_language = st.selectbox("Select Language for Speech", list(language_options.keys()))

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.subheader("Model Generated Legal and Forensic Evidence Caption")
        st.write("🔄 Generating caption...")
        caption2 = generate_caption_2(image)
        caption = rephrase_caption(caption2)
        st.success(f"Caption: {caption}")

        # Sidebar for Case Evidence Prompt
        st.sidebar.title("🖼️ Legal Forensic Support")
        st.sidebar.header("Case Evidence Inquiry")
        prompt_input = st.sidebar.text_area("Enter your prompt for case evidence:")
        if prompt_input:
            st.sidebar.write("**AI Response:**")
            response = gemini_model.generate_content(prompt_input)
            st.sidebar.write(response.text)

        if st.button("🔊 Convert to Speech"):
            text_to_speech(caption, language_options[selected_language])
