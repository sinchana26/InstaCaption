from gtts import gTTS
from playsound import playsound
from translate import Translator  # More reliable translation library
import os


def text_to_speech():
    languages = {
        "1": ("English", "en"),
        "2": ("Kannada", "kn"),
        "3": ("Hindi", "hi"),
        "4": ("Telugu", "te"),
        "5": ("Tamil", "ta"),
    }

    print("Select a language option:")
    for key, (lang_name, _) in languages.items():
        print(f"{key}. {lang_name}")

    choice = input("Enter your choice (1-5): ")

    if choice not in languages:
        print("Invalid choice. Please select a valid option.")
        return

    lang_name, lang_code = languages[choice]
    text = input("Enter the text you want to convert to speech: ")

    try:
        # Use alternative translation library
        translator = Translator(to_lang=lang_code)
        translated_text = translator.translate(text)
        print(f"\nTranslated Text in {lang_name}: {translated_text}")

        # Convert translated text to speech
        tts = gTTS(text=translated_text, lang=lang_code)
        filename = "output.mp3"
        tts.save(filename)

        print("\n🔊 Playing the generated speech...")
        playsound(filename)

        # Remove the file after playing
        os.remove(filename)

    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    text_to_speech()
