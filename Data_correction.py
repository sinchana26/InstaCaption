import re

# Input and Output file paths
input_file = "captions.txt"
output_file = "cleaned_captions.txt"

# Function to remove punctuation but keep first comma
def clean_caption(line):
    parts = line.split(",", 1)  # Split only at the first comma
    if len(parts) == 2:
        image_name, caption = parts[0], parts[1]
        caption = re.sub(r'[^\w\s]', '', caption)  # Remove punctuation
        return f"{image_name},{caption.strip()}\n"
    return line  # If the format is incorrect, return it unchanged

# Read, clean, and write to a new file
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        cleaned_line = clean_caption(line)
        outfile.write(cleaned_line)

print("✅ Processing Complete! Cleaned captions saved in 'cleaned_captions.txt'.")

