import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class ImageCaptionGenerator:
    def __init__(self, model_name="Salesforce/blip-image-captioning-large"):
        """Initialize the processor and model for image captioning."""
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

    def generate_caption(self, image_path):
        """Generates a caption for the given image."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Invalid image path: {image_path}")

        image = Image.open(image_path).convert("RGB")
        text = "You are seeing a"
        inputs = self.processor(image, text, return_tensors="pt")
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        return caption


if __name__ == "__main__":
    image_path = "Evaluation/Crime_136.jpg"
    caption_generator = ImageCaptionGenerator()
    try:
        caption = caption_generator.generate_caption(image_path)
        print(f"Caption: {caption}")
    except FileNotFoundError as e:
        print(e)
